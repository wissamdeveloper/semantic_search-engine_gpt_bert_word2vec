import pickle
from flask import Blueprint, Flask, render_template, request, jsonify
import nltk
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
import faiss
import sqlite3
import os

import torch

# Define the path to save the .index file
index_file_path = "my_index.index"
dimension = 768

bertsearch = Blueprint("bertsearch", __name__, template_folder="templates")

def get_bert_embedding(text):
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)

    # Encode the text to obtain embeddings
    embeddings = model.encode(text, convert_to_tensor=True)

    # Convert the tensor to a numpy array
    embedding_array = embeddings.cpu().numpy()

    return embedding_array

@bertsearch.route('/bertsearch', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Get the user's prompt from the form
        user_prompt = request.form.get('prompt')
        print("user_prompt", user_prompt)
        # Compute the embedding for the user's prompt
        prompt_embedding = get_bert_embedding(user_prompt)
        print("prompt_embedding", prompt_embedding)

        if os.path.exists(index_file_path):
            # If the index file exists, read it
            faiss_index = faiss.read_index(index_file_path)
        else:
            # If the index file does not exist, create and populate the index
            faiss_index = faiss.IndexFlatL2(dimension)
            # Populate the index with your data (You need to add this logic)

        # Perform a similarity search using the Faiss index
        k = 5  # Number of nearest neighbors to retrieve
        print("faiss_index", faiss_index)
        distances, indices = faiss_index.search(np.array([prompt_embedding]), k)
        print("indices", indices)
        # Extract the metadata for the top retrieved index
        top_index = indices[0][0]
        top_index = top_index + 1
        print("top_index")
        conn = sqlite3.connect('metadata.db')

        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Specify the id you want to retrieve
        target_id = int(top_index)  # Change this to the desired id
        print("Data type of top_index:", type(top_index))

        # Execute an SQL query with a WHERE clause to retrieve data for a specific id
        cursor.execute("SELECT * FROM bertmetadata WHERE id=?", (target_id,))

        # Fetch the first row from the result set (assuming id is unique)
        row = cursor.fetchone()

        print("row", row)
        cursor.close()
        conn.close()

        if row:
            id, top_file_name, top_page_number, top_chunk_number, top_chunk, top_embedding = row
            top_chunk_number = int(top_chunk_number)  # Convert to integer
             # Deserialize the embedding from the byte string
            top_embedding = pickle.loads(top_embedding)

            # Convert the deserialized embedding back to a tensor
            top_embedding = torch.from_numpy(top_embedding)

            # Convert the tensor to a NumPy array and then extract the values
            top_embedding_values = top_embedding.numpy()

            # Now, top_embedding_values is a NumPy array with the embedding values
            print("Type of top_embedding_values:", type(top_embedding_values))
            # Reshape the embeddings into NumPy arrays
            embedding1 = np.array(prompt_embedding).reshape(1, -1)  # Reshape to a row vector
            embedding2 = np.array(top_embedding).reshape(1, -1)  # Reshape to a row vector
            embedding1 = torch.from_numpy(embedding1)
            embedding2 = torch.from_numpy(embedding2)

            # Calculate the cosine similarity
            similarity = torch.cosine_similarity(embedding1, embedding2)
            cosine_similarity = similarity.item()

            print("Cosine Similarity:", cosine_similarity)

            # print("Cosine Similarity:", similarity[0][0])

            # top_embedding = int(top_embedding)  # Assuming top_embedding is a utf-8 encoded string
            # You can now use top_file_name, top_page_number, and top_chunk_number
            # to identify and display the corresponding chunk to the user
            response = {
                "file_name": top_file_name,
                "page_number": top_page_number + 1,
                "chunk_number": top_chunk_number,
                "top_chunk": top_chunk,
                "top_embedding": top_embedding.tolist()
            }
            return jsonify(response)
        else:
            return "No matching document found."
    return render_template('bertsearch.html')

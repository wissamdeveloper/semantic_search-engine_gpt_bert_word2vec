from flask import Flask, render_template, request, jsonify
import nltk
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
import faiss
import sqlite3
import os
import torch
import openai  # Import the OpenAI library
import transformers
from transformers import GPT2Tokenizer

nltk.download('punkt')

app = Flask(__name__)

# Define the path to save the .index file
index_file_path = "my_index.index"

# Initialize Faiss index and load it from the .index file
dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index = faiss.read_index(index_file_path)

# Set up your OpenAI API key
# Replace 'your-api-key' with your actual OpenAI API key
openai.api_key = "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"

# Load the GPT-2 model tokenizer
gpt2_model_name = "gpt2"  # You can choose a different GPT-2 variant if needed
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

def get_bert_embedding(text):
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)

    # Encode the text to obtain embeddings
    embeddings = model.encode(text, convert_to_tensor=True)

    # Convert the tensor to a numpy array
    embedding_array = embeddings.cpu().numpy()

    return embedding_array

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Get the user's prompt from the form
        user_prompt = request.form.get('prompt')

        # Compute the embedding for the user's prompt
        prompt_embedding = get_bert_embedding(user_prompt)

        # Perform a similarity search using the Faiss index
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = faiss_index.search(np.array([prompt_embedding]), k)

        # Extract the metadata for the top retrieved index
        top_index = indices[0][0]
        top_index = top_index + 1

        conn = sqlite3.connect('metadata.db')
        cursor = conn.cursor()

        # Specify the id you want to retrieve
        target_id = int(top_index)
        cursor.execute("SELECT * FROM metadata WHERE id=?", (target_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            id, top_file_name, top_page_number, top_chunk_number, top_chunk = row
            top_chunk_number = int(top_chunk_number)

            # Use GPT-3 to generate words based on top_chunk
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=top_chunk,
                max_tokens=100,
                n=1
            )

            generated_text = response.choices[0].text

            response = {
                "file_name": top_file_name,
                "page_number": top_page_number + 1,
                "chunk_number": top_chunk_number,
                "generated_text": generated_text  # Include generated text in the response
            }
            return jsonify(response)
        else:
            return "No matching document found."
    return render_template('search.html')

if __name__ == "__main__":
    app.run(port=5000)

import nltk
import os
from flask import Blueprint, Flask, render_template, request, jsonify
import numpy as np
import faiss
import sqlite3
import re
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch

app = Flask(__name__)


# Specify the NLTK data directory explicitly as a list of paths
home_directory = os.path.expanduser("~")

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Define the path to save the .index file
index_file_path = "my_index.index"
# Check if the index file exists
if not os.path.exists(index_file_path):
    print("Index file not found. Please upload files first.")
    # You can return a message here or render a specific template to display the message
else:
    # Initialize Faiss index and load it from the .index file
    dimension = 300  # Adjust the dimension based on your Word2Vec model
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index = faiss.read_index(index_file_path)

# Combine the paths to create the full model_file_path
downloads_folder = "Downloads"
model_filename = "word2vec-google-news-300.gz"
model_file_path = os.path.join(home_directory, downloads_folder, model_filename)

try:
    # Load the Word2Vec model from the downloaded file
    word2vec_model = KeyedVectors.load_word2vec_format(model_file_path, binary=True)
except FileNotFoundError:
    print(f"Model file not found at '{model_file_path}'.")

# Initialize NLTK's stopwords and WordNet lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# def download_nltk_resources():
#     try:
#         # Download NLTK stopwords if not already downloaded
#         nltk.data.find("corpora/stopwords.zip")
#     except LookupError:
#         nltk.download('stopwords')

# @app.before_first_request
# def before_first_request():
#     # Perform one-time setup before the first request to the application
#     download_nltk_resources()

def preprocess_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove punctuation and convert to lowercase
    words = [re.sub(r'[^\w\s]', '', word).lower() for word in words]

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

def get_word2vec_embedding(text):
    # Preprocess the text
    preprocessed_text = preprocess_text(text)

    # Tokenize the preprocessed text into words
    words = nltk.word_tokenize(preprocessed_text)

    # Initialize an empty list to store word vectors
    word_vectors = []

    # Compute the Word2Vec embeddings for each word
    for word in words:
        try:
            word_vector = word2vec_model[word]
            print("word_vector", word_vector)
            word_vectors.append(word_vector)
        except KeyError:
            # Handle the case where a word is not in the vocabulary
            pass

    # Convert the list of word vectors to a numpy array
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average the word vectors
    else:
        return np.zeros(dimension)  # Return a zero vector if no valid words are found

@app.route('/word2vecsearch', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        # Get the user's prompt from the form
        user_prompt = request.form.get('prompt')

        # Compute the embedding for the user's prompt using Word2Vec
        prompt_embedding = get_word2vec_embedding(user_prompt)
        print("prompt_embedding", prompt_embedding)
        # Perform a similarity search using the Faiss index
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = faiss_index.search(np.array([prompt_embedding]), k)
        print("indices", indices)
        # Extract the metadata for the top retrieved index
        top_index = indices[0][0]
        top_index = int(top_index) + 1  # Adjust for SQLite's 1-based indexing

        conn = sqlite3.connect('metadata.db')
        cursor = conn.cursor()

        # Execute an SQL query with a WHERE clause to retrieve data for a specific id
        cursor.execute("SELECT * FROM metadata WHERE id=?", (top_index,))

        # Fetch the first row from the result set (assuming id is unique)
        row = cursor.fetchone()
        print("row", row)
        cursor.close()
        conn.close()

        if row:
            id, top_file_name, top_page_number, top_chunk_number, top_chunk_text, top_chunk_embedding = row
            # Convert top_chunk_embedding_str to a numeric NumPy array
            top_chunk_embedding = np.fromstring(top_chunk_embedding[1:-1], sep=' ')
            
            # Ensure it's a float64 array (change dtype if needed)
            top_chunk_embedding = top_chunk_embedding.astype(np.float64)

            # Reshape the embeddings into NumPy arrays
            embedding1 = np.array(prompt_embedding).reshape(1, -1)  # Reshape to a row vector
            embedding2 = top_chunk_embedding.reshape(1, -1)  # Reshape to a row vector
            embedding1 = torch.from_numpy(embedding1)
            embedding2 = torch.from_numpy(embedding2)

            # Calculate the cosine similarity
            similarity = torch.cosine_similarity(embedding1, embedding2)
            cosine_similarity = similarity.item()

            print("Cosine Similarity:", cosine_similarity)
            response = {
                "file_name": top_file_name,
                "page_number": top_page_number + 1,
                "chunk_number": top_chunk_number,
                "chunk_text": top_chunk_text,
                "top_chunk_embedding": top_chunk_embedding.tolist(),  # Convert back to a list for JSON serialization
            }
            return jsonify(response)
        else:
            return "No matching document found."
    
    return render_template('word2vecsearch.html')
if __name__ == "__main__":
    app.run(debug=True)

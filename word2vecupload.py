from flask import Blueprint, Flask, render_template, request
import nltk
from gensim.models import Word2Vec
import fitz
import numpy as np
from io import BytesIO
import faiss
import os
import sqlite3
from gensim.models import KeyedVectors  # Import KeyedVectors
import re 
from concurrent.futures import ThreadPoolExecutor


word2vecupload = Blueprint("word2vecupload", __name__, template_folder="templates")

nltk.download('punkt')

index_file_path = "my_index.index"
database_file_path = "metadata.db"

dimension = 300  # Adjust the dimension based on your Word2Vec model
faiss_index = faiss.IndexFlatL2(dimension)

conn = sqlite3.connect(database_file_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY,
        filename TEXT,
        pagenumber INTEGER,
        chunknumber INTEGER,
        chunktext TEXT,
        embedding TEXT  -- Add a new column to store embeddings as text
    )
''')
conn.commit()
conn.close()

home_directory = os.path.expanduser("~")

# Specify the relative path to the Downloads folder and the model file
downloads_folder = "Downloads"
model_filename = "word2vec-google-news-300.gz"

# Combine the paths to create the full model_file_path
model_file_path = os.path.join(home_directory, downloads_folder, model_filename)

try:
    # Load the Word2Vec model from the downloaded file
    word2vec_model = KeyedVectors.load_word2vec_format(model_file_path, binary=True)
except FileNotFoundError:
    print(f"Model file not found at '{model_file_path}'.")

def get_word2vec_embedding(text, model):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove punctuation and convert to lowercase
    words = [re.sub(r'[^\w\s]', '', word).lower() for word in words]

    # Initialize an empty list to store word vectors
    word_vectors = []

    # Compute the word embeddings for each word
    for word in words:
        try:
            word_vector = model[word]
            word_vectors.append(word_vector)
        except KeyError:
            # Handle the case where a word is not in the vocabulary
            pass

    # Convert the list of word vectors to a numpy array
    if word_vectors:
        return np.mean(word_vectors, axis=0)  # Average the word vectors
    else:
        return np.zeros(model.vector_size)  # Return a zero vector if no valid words are found

def convert_embedding_to_string(embedding):
    # Convert the embedding numpy array to a string
    return ' '.join(map(str, embedding))

def store_embedding(embedding, file_name, page_number, chunk_number, chunk_text):
    # Check if the dimension of the embedding matches the FAISS index dimension
    if len(embedding) != dimension:
        print(f"Warning: Embedding dimension ({len(embedding)}) does not match the FAISS index dimension ({dimension}). Skipping this embedding.")
        return
     # Convert the embedding to a float32 array
    embedding = embedding.astype('float32')

    # Convert the embedding to a string format
    embedding_str = convert_embedding_to_string(embedding)

    faiss_index.add(np.array([embedding]))

    conn = sqlite3.connect(database_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO metadata (filename, pagenumber, chunknumber, chunktext, embedding)
        VALUES (?, ?, ?, ?, ?)
    ''', (file_name, page_number, chunk_number, chunk_text, embedding_str))
    conn.commit()
    conn.close()

@word2vecupload.route('/word2vecupload', methods=['GET', 'POST'])
def index():
    uploaded_file = request.files.get('pdf_file')
    if request.method == 'POST':
        unique_id = 1
        if uploaded_file:
            pdf_data = BytesIO(uploaded_file.read())
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            # Create a thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    pagetext = page.get_text()
                    words = nltk.word_tokenize(pagetext)
                    chunk_size = 50
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i + chunk_size])
                        futures.append(executor.submit(process_chunk, chunk, uploaded_file.filename, page_num))

                # Wait for all futures to complete
                for future in futures:
                    future.result()

            # Save the FAISS index (if needed)
            faiss.write_index(faiss_index, index_file_path)

            return "PDF file uploaded and processed."
        else:
            return "No PDF file uploaded."
    return render_template('index.html')
def process_chunk(chunk, filename, page_num):
    chunk_embedding = get_word2vec_embedding(chunk, word2vec_model)

    # Check if the dimension of the embedding matches before storing
    if len(chunk_embedding) == dimension:
        store_embedding(
            chunk_embedding,
            filename,
            page_num,
            chunk
        )
    else:
        print(f"Warning: Embedding dimension ({len(chunk_embedding)}) does not match the FAISS index dimension ({dimension}). Skipping this embedding.")

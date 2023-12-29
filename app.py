from flask import Flask, render_template, request, jsonify
import nltk
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
import faiss
import os
import sqlite3
from gensim.models import Word2Vec, KeyedVectors
import re

nltk.download('punkt')

app = Flask(__name__)

# Define paths and configurations
index_file_path_bert = "my_index_bert.index"
index_file_path_word2vec = "my_index_word2vec.index"
database_file_path = "metadata.db"
downloads_folder = "Downloads"
model_filename = "word2vec-google-news-300.gz"
dimension_bert = 768
dimension_word2vec = 300

# Initialize Faiss indices and load them from files
faiss_index_bert = faiss.IndexFlatL2(dimension_bert)
faiss_index_bert = faiss.read_index(index_file_path_bert)
faiss_index_word2vec = faiss.IndexFlatL2(dimension_word2vec)
faiss_index_word2vec = faiss.read_index(index_file_path_word2vec)

# Load the Word2Vec model from the downloaded file
home_directory = os.path.expanduser("~")
model_file_path = os.path.join(home_directory, downloads_folder, model_filename)
try:
    word2vec_model = KeyedVectors.load_word2vec_format(model_file_path, binary=True)
except FileNotFoundError:
    print(f"Model file not found at '{model_file_path}'.")

# Initialize NLTK's stopwords and WordNet lemmatizer
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [re.sub(r'[^\w\s]', '', word).lower() for word in words]
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def get_bert_embedding(text):
    model_name = 'all-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text, convert_to_tensor=True)
    embedding_array = embeddings.cpu().numpy()
    return embedding_array

def get_word2vec_embedding(text):
    preprocessed_text = preprocess_text(text)
    words = nltk.word_tokenize(preprocessed_text)
    word_vectors = []
    for word in words:
        try:
            word_vector = word2vec_model[word]
            word_vectors.append(word_vector)
        except KeyError:
            pass
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(dimension_word2vec)

def store_bert_embedding(embedding, file_name, page_number, chunk_number, chunk_text):
    faiss_index_bert.add(np.array([embedding]))
    conn = sqlite3.connect(database_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO metadata (file_name, page_number, chunk_number, chunk_text)
        VALUES (?, ?, ?, ?)
    ''', (file_name, page_number, chunk_number, chunk_text))
    conn.commit()
    conn.close()

def store_word2vec_embedding(embedding, file_name, page_number, chunk_number, chunk_text):
    faiss_index_word2vec.add(np.array([embedding]))
    conn = sqlite3.connect(database_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO metadata (file_name, page_number, chunk_number, chunk_text)
        VALUES (?, ?, ?, ?)
    ''', (file_name, page_number, chunk_number, chunk_text))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('pdf_file')
        if uploaded_file:
            pdf_data = BytesIO(uploaded_file.read())
            unique_id_bert = 1
            unique_id_word2vec = 1
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pagetext = page.get_text()
                words = nltk.word_tokenize(pagetext)
                chunk_size = 50
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    
                    # Calculate embeddings and store them in respective indices
                    bert_embedding = get_bert_embedding(chunk)
                    word2vec_embedding = get_word2vec_embedding(chunk)
                    
                    store_bert_embedding(
                        bert_embedding,
                        uploaded_file.filename,
                        page_num,
                        unique_id_bert,
                        chunk
                    )
                    
                    store_word2vec_embedding(
                        word2vec_embedding,
                        uploaded_file.filename,
                        page_num,
                        unique_id_word2vec,
                        chunk
                    )
                    
                    unique_id_bert += 1
                    unique_id_word2vec += 1

            faiss.write_index(faiss_index_bert, index_file_path_bert)
            faiss.write_index(faiss_index_word2vec, index_file_path_word2vec)

            return "PDF file uploaded and processed."
        else:
            return "No PDF file uploaded."
    return render_template('index.html')

@app.route('/search_bert', methods=['POST'])
def search_bert():
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        prompt_embedding = get_bert_embedding(user_prompt)
        k = 5
        distances, indices = faiss_index_bert.search(np.array([prompt_embedding]), k)
        top_index = indices[0][0]
        top_index = int(top_index) + 1
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metadata WHERE id=?", (top_index,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            id, top_file_name, top_page_number, top_chunk_number, top_chunk_text = row
            response = {
                "file_name": top_file_name,
                "page_number": top_page_number + 1,
                "chunk_number": top_chunk_number,
                "chunk_text": top_chunk_text
            }
            return jsonify(response)
        else:
            return "No matching document found."

@app.route('/search_word2vec', methods=['POST'])
def search_word2vec():
    if request.method == 'POST':
        user_prompt = request.form.get('prompt')
        prompt_embedding = get_word2vec_embedding(user_prompt)
        k = 5
        distances, indices = faiss_index_word2vec.search(np.array([prompt_embedding]), k)
        top_index = indices[0][0]
        top_index = int(top_index) + 1
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM metadata WHERE id=?", (top_index,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            id, top_file_name, top_page_number, top_chunk_number, top_chunk_text = row
            response = {
                "file_name": top_file_name,
                "page_number": top_page_number + 1,
                "chunk_number": top_chunk_number,
                "chunk_text": top_chunk_text
            }
            return jsonify(response)
        else:
            return "No matching document found."

if __name__ == "__main__":
    app.run(port=5000)

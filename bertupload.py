import pickle
from flask import Blueprint, Flask, render_template, request
import nltk
from sentence_transformers import SentenceTransformer
import fitz
import numpy as np
from io import BytesIO
import faiss
import os
import sqlite3
import multiprocessing

nltk.download('punkt')
app = Flask(__name__)
bertupload = Blueprint("bertupload", __name__, template_folder="templates")

model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

def create_database_if_not_exists(database_file_path):
    if not os.path.exists(database_file_path):
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bertmetadata (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                page_number INTEGER,
                chunk_number INTEGER,
                chunk_text TEXT,
                embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()

create_database_if_not_exists("metadata.db")

dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)

def get_bert_embedding(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    embedding_array = embeddings.cpu().numpy()
    return embedding_array

def store_embedding_batch(batch_embeddings, batch_metadata):
    faiss_index.add(np.array(batch_embeddings))
    conn = sqlite3.connect("metadata.db")
    cursor = conn.cursor()
    for i in range(len(batch_embeddings)):
        # Serialize the embedding using pickle
        embedding_blob = pickle.dumps(batch_embeddings[i])
        metadata = batch_metadata[i] + (embedding_blob,)
        print(f"Type of embedding: {type(batch_embeddings[i])}")

        cursor.execute('''
            INSERT INTO bertmetadata (file_name, page_number, chunk_number, chunk_text, embedding)
            VALUES (?, ?, ?, ?, ?)
        ''', metadata)
    conn.commit()
    conn.close()

@bertupload.route('/api/bertupload', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files.get('pdf_file')
        print("uploaded_file", uploaded_file)
        if uploaded_file:
            print("uploaded_file",uploaded_file)
            pdf_data = BytesIO(uploaded_file.read())
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            chunk_embeddings = []
            chunk_metadata = []
            unique_id = 1

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pagetext = page.get_text()
                words = nltk.word_tokenize(pagetext)
                chunk_size = 50
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunk_embedding = get_bert_embedding(chunk)
                    chunk_embeddings.append(chunk_embedding)
                    chunk_metadata.append((uploaded_file.filename, page_num, unique_id, chunk))
                    unique_id += 1

            # Batch processing of embeddings and metadata
            store_embedding_batch(chunk_embeddings, chunk_metadata)

            faiss.write_index(faiss_index, "my_index.index")

            return "PDF file uploaded and processed."
        else:
            return "No PDF file uploaded."
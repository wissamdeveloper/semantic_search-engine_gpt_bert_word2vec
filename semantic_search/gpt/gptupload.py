from flask import Flask, render_template, request
import nltk
import fitz
import numpy as np
from io import BytesIO
import faiss
import os
import sqlite3  # Import sqlite3 module
from transformers import GPT2Tokenizer, GPT2Model
import torch

nltk.download('punkt')

app = Flask(__name__)

def create_database_if_not_exists():
    if not os.path.exists(database_file_path):
        conn = sqlite3.connect(database_file_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                page_number INTEGER,
                chunk_number INTEGER,
                chunk_text TEXT
            )
        ''')
        conn.commit()
        conn.close()

index_file_path = "my_index.index"
database_file_path = "metadata.db"  # SQLite database file path

dimension = 768  # Adjust the dimension based on the GPT-2 model you choose
faiss_index = faiss.IndexFlatL2(dimension)

# Create a SQLite database and a table to store metadata
conn = sqlite3.connect(database_file_path)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY,
        file_name TEXT,
        page_number INTEGER,
        chunk_number INTEGER,
        chunk_text TEXT
    )
''')
conn.commit()
conn.close()

# Load the GPT-2 model and tokenizer (replace 'gpt2' with the model name you want to use)
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

def get_gpt2_embedding(text):
    # Tokenize the text and obtain hidden states from GPT-2
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def store_embedding(embedding, file_name, page_number, chunk_number, chunk_text):
    faiss_index.add(np.array([embedding]))
    
    # Insert metadata into the SQLite database
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
    uploaded_file = request.files.get('pdf_file')
    if request.method == 'POST':
        if uploaded_file:
            pdf_data = BytesIO(uploaded_file.read())
            unique_id = 1
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text()
                words = nltk.word_tokenize(page_text)
                chunk_size = 100
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunk_embedding = get_gpt2_embedding(chunk)
                    store_embedding(
                        chunk_embedding,
                        uploaded_file.filename,
                        page_num,
                        unique_id,
                        chunk
                    )
                    unique_id += 1

            faiss.write_index(faiss_index, index_file_path)

            return "PDF file uploaded and processed."
        else:
            return "No PDF file uploaded."
    return render_template('index.html')

if __name__ == "__main__":
    create_database_if_not_exists()  # Create the database if it doesn't exist
    app.run(port=5001)

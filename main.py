from flask import Blueprint, Flask, render_template, request, jsonify
from flask_cors import CORS
from bertupload import bertupload
from bertsearch import bertsearch
# from word2vecsearch import word2vecsearch
from word2vecupload import word2vecupload
import numpy as np
from scipy.spatial.distance import cosine
app = Flask(__name__)
app.register_blueprint(bertupload)
app.register_blueprint(bertsearch)
app.register_blueprint(word2vecupload)
# app.register_blueprint(word2vecsearch)

if __name__ == "__main__":
    CORS(app)
    app.run(debug=True)

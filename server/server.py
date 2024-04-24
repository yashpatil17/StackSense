from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.simplefilter("ignore")

app = Flask(__name__)
CORS(app, origins="*")  

# Load the trained model
clf = joblib.load('../models/linearSVC.pkl')

# Load the vectorizer used for preprocessing during training
vectorizer = joblib.load('../vectorizer/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    return ['Hello', 'There']

if __name__ == '__main__':
    app.run(debug=False)

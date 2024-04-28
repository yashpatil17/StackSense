from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

warnings.simplefilter("ignore")

app = Flask(__name__)
CORS(app, origins="*")  

# Load the trained model
clf = joblib.load('../models/linearSVC.pkl')
mlb = joblib.load('../models/mlb_weights.pkl')

# Load the vectorizer used for preprocessing during training
vectorizer = joblib.load('../vectorizer/tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Preprocess text
    cleaned_text = cleaning(data['input_data'])
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    # Make prediction
    prediction = clf.predict(text_vectorized)
    print(prediction)
    predicted_tags = mlb.inverse_transform(prediction)
    print(predicted_tags)
    # Example response
    response = {
        'prediction': list(predicted_tags) 
    }
    
    return jsonify(response)

def cleaning(text):
    text = text.lower()
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    clean = re.compile('<.*?>')
    text = re.sub(clean,'',text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    text = ' '.join(tokens)
    return text

if __name__ == '__main__':
    app.run(debug=False)

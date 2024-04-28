from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from scipy.sparse import hstack
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app, origins="*")

# Load the trained model
clf = joblib.load('../models/linearSVC.pkl')
mlb = joblib.load('../models/mlb_weights.pkl')
vectorizer = joblib.load('../vectorizer/tfidf_vectorizer.pkl')

word2vec_model = Word2Vec.load("../embeddings/word2vec_model.bin")
tfidf_embeddings = joblib.load('../embeddings/tfidf_vectorizer.pkl')
loaded_use_module = tf.saved_model.load("../embeddings/use_model")

data_path = "../output/df_eda.pkl"

data_path_unprocessed = "../df_pre.pkl"

df_unprocessed = pd.read_pickle(data_path_unprocessed)

df = pd.read_pickle(data_path)

# Load KNN model
with open("../embeddings/knn_model.pkl", "rb") as f:
    knn_model = joblib.load(f)
with open("../embeddings/use_knn_model.pkl", "rb") as f:
    use_knn_model = joblib.load(f)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Preprocess text
    cleaned_text = cleaning(data['input_data'])
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    # Make prediction
    prediction = clf.predict(text_vectorized)
    predicted_tags = mlb.inverse_transform(prediction)
    
    # Example response
    response = {
        'prediction': list(predicted_tags) 
    }
    
    return jsonify(response)

@app.route('/similar_questions', methods=['POST'])
def similar_questions():
    data = request.get_json(force=True)
    input_question = data['input_data']
    selected_vectorizer=data['vectorizer']
    input_tfidf = tfidf_embeddings.transform([input_question])

    if selected_vectorizer=="tfidf+word2vec":
        input_word2vec = document_embedding(input_question, word2vec_model)
        print("embed")
        print(input_word2vec)
    
        if input_word2vec is None:
            return jsonify({"error": "Input question not in vocabulary."}), 400
    
        input_embeddings = hstack([input_tfidf, input_word2vec.reshape(1, -1)])
    
        # Find nearest neighbors
        _, indices = knn_model.kneighbors(input_embeddings)

    elif selected_vectorizer == "universal_sentence_embedding":

        input_use = document_embedding_use(input_question, loaded_use_module)
        if input_use is None:
            return jsonify({"error": "Input question not in vocabulary."}), 400
    
        input_embeddings_use = hstack([input_tfidf, input_use.reshape(1, -1)])
        
        _, indices = use_knn_model.kneighbors(input_embeddings_use)
    # Extract similar questions
    similar_questions_indices = indices[0][1:]  # Exclude the input question itself
    similar_questions = df_unprocessed.iloc[similar_questions_indices]['Title']
    similar_questions_body = df_unprocessed.iloc[similar_questions_indices]['Body']

    response = {
        'title': list(similar_questions),
        'body': list(similar_questions_body)
    }
    # Return similar questions
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

def document_embedding_use(text, use_module):
    embeddings = use_module([text])
    return np.array(embeddings).squeeze()

def document_embedding(text, model):
    # Tokenize words
    tokenized_text = simple_preprocess(text)
    # Filter out words not in vocabulary
    words_in_vocab = [word for word in tokenized_text if word in model.wv.key_to_index]
    # If no words in vocabulary, return None
    if not words_in_vocab:
        return None
    # Calculate mean of word vectors
    return sum(model.wv[word] for word in words_in_vocab) / len(words_in_vocab)

if __name__ == '__main__':
    app.run(debug=False)

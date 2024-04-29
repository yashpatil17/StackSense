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
# import tensorflow as tf
import torch
from transformers import BertTokenizer
import numpy as np

import sys
sys.path.append('../')
from models.bert2 import BERTClass

app = Flask(__name__)
CORS(app, origins="*")

# Load the trained model
clf = joblib.load('../models/linearSVC.pkl')
mlb = joblib.load('../models/mlb_weights.pkl')
vectorizer = joblib.load('../vectorizer/tfidf_vectorizer.pkl')

word2vec_model = Word2Vec.load("../embeddings/word2vec_model.bin")
tfidf_embeddings = joblib.load('../embeddings/tfidf_vectorizer.pkl')
# loaded_use_module = tf.saved_model.load("../embeddings/use_model")

data_path = "../output/df_eda.pkl"

data_path_unprocessed = "../df_pre.pkl"

df_unprocessed = pd.read_pickle(data_path_unprocessed)

df = pd.read_pickle(data_path)

# Load KNN model
with open("../embeddings/knn_model.pkl", "rb") as f:
    knn_model = joblib.load(f)
with open("../embeddings/use_knn_model.pkl", "rb") as f:
    use_knn_model = joblib.load(f)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the trained BERT model
model = BERTClass()
# model = torch.nn.DataParallel(model)

checkpoint = torch.load('../models/bert.pt')
state_dict = checkpoint['state_dict']

# Remove the "module." prefix from keys (if present)
# if not next(iter(state_dict.keys())).startswith('module'):
#     state_dict = {f'module.{k}': v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Preprocess text
    cleaned_text = cleaning(data['input_data'])

    #### Normal model prediction
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    # Make prediction
    prediction = clf.predict(text_vectorized)
    predicted_tags = mlb.inverse_transform(prediction)
    print("LinearSVC prediction:", predicted_tags)


    ##### BERT model prediction
    # Tokenize and preprocess the question
    inputs = tokenizer.encode_plus(cleaned_text, None, add_special_tokens=True, max_length=256, padding='max_length', return_token_type_ids=True, truncation=True, return_tensors='pt')

    targets = ['.net', 'ajax', 'algorithm', 'android', 'angularjs', 'api', 'arrays', 'asp.net', 'asp.net-mvc', 'asp.net-mvc-3', 'bash', 'c', 'c#', 'c++', 'c++11', 'cocoa', 'cocoa-touch', 'css', 'css3', 'database', 'datetime', 'debugging', 'delphi', 'django', 'eclipse', 'emacs', 'entity-framework', 'exception', 'facebook', 'function', 'gcc', 'generics', 'git', 'google-chrome', 'haskell', 'hibernate', 'html', 'html5', 'http', 'image', 'ios', 'ipad', 'iphone', 'java', 'javascript', 'jquery', 'json', 'linq', 'linux', 'list', 'math', 'matlab', 'maven', 'mongodb', 'multithreading', 'mysql', 'node.js', 'numpy', 'objective-c', 'oop', 'optimization', 'oracle', 'osx', 'performance', 'perl', 'php', 'postgresql', 'python', 'qt', 'r', 'regex', 'rest', 'ruby', 'ruby-on-rails', 'ruby-on-rails-3', 'scala', 'security', 'shell', 'spring', 'sql', 'sql-server', 'sql-server-2008', 'string', 'svn', 'swift', 'swing', 'templates', 'tsql', 'twitter-bootstrap', 'unit-testing', 'vb.net', 'vim', 'visual-studio', 'visual-studio-2010', 'wcf', 'windows', 'winforms', 'wpf', 'xcode', 'xml']

    # Make prediction
    with torch.no_grad():
        outputs = model(ids=inputs['input_ids'], mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])

        # ids = data['ids'].to(device, dtype = torch.long)
        # mask = data['mask'].to(device, dtype = torch.long)
        # token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        # targets = data['targets'].to(device, dtype = torch.float)
        # outputs = model(ids, mask, token_type_ids)
        # y_test.extend(targets.cpu().detach().numpy().tolist())
        # y_pred.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        # print(outputs)

        # predicted_labels = torch.sigmoid(outputs)
        # print(predicted_labels)

        # y_pred = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        # print(y_pred)

        # y_pred = (np.array(y_pred) > np.mean(y_pred)).astype(int)
        # print(y_pred)

        # y_pred = (np.array(y_pred) > 0.5).astype(int)
        # print(y_pred)


    # predicted_tags = []
    # indices = np.where(y_pred == 1)[1]
    # print(indices)

    # for i in indices:
    #     print(targets[i])
    #     predicted_tags.append(targets[i])


    k = 5  # Choose the top k predictions

    # Get the top k values and indices
    top_values, top_indices = torch.topk(outputs, k)

    # Print the corresponding target labels for the top k predictions
    print("Top {} predicted labels:".format(k))
    for i in range(k):
        print("{}. {}".format(i+1, targets[top_indices[0][i].item()]))

    # Store the top predicted tags in a list
    predicted_tags = [targets[top_indices[0][i].item()] for i in range(k)]

    print(predicted_tags)

    print("BERT CLEANED PREDICTED TAGS")
    cleaned_predicted_tags = [predicted_tags[0]] + [tag for tag in predicted_tags[1:] if tag in cleaned_text]
    for tag in cleaned_predicted_tags:
        print(tag)

    print(cleaned_predicted_tags)

    predicted_tags = cleaned_predicted_tags

    # Find the index of the maximum value in the predictions tensor
    # max_index = torch.argmax(outputs)

    # # Print the corresponding target label
    # print("Predicted label:", targets[max_index.item()])

    # predicted_tags = []
    # predicted_tags.append(targets[max_index.item()])
    # max_indices = torch.nonzero(outputs == torch.max(outputs)).flatten()
    # # Print the corresponding target labels for each index
    # print("Predicted labels:")
    # for index in max_indices:
    #     print(targets[index.item()])

        
    # # predicted_tags = mlb.inverse_transform(predd)
    # predicted_tags = []

    # # Iterate over each prediction vector
    # for pred_vector in predd:
    #     # Find the indices where the value is 1
    #     indices = torch.nonzero(pred_vector).flatten()
    #     # Print the corresponding target labels
    #     print("Predicted labels:")
    #     for index in indices:
    #         print(targets[index.item()])
    #         predicted_tags.append(targets[index.item()])

    # Convert predicted labels to list
    # predicted_labels = predicted_labels.cpu().numpy().tolist()

    # Assuming your labels are binary, you can apply a threshold to convert probabilities to binary predictions
    # predicted_tags = [1 if pred >= 0.5 else 0 for pred in predicted_labels[0]]

    
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

    # elif selected_vectorizer == "universal_sentence_embedding":

    #     input_use = document_embedding_use(input_question, loaded_use_module)
    #     if input_use is None:
    #         return jsonify({"error": "Input question not in vocabulary."}), 400
    
    #     input_embeddings_use = hstack([input_tfidf, input_use.reshape(1, -1)])
        
    #     _, indices = use_knn_model.kneighbors(input_embeddings_use)
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

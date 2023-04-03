# import necessary libraries
import numpy as np
import pandas as pd
import gensim
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load pre-trained Word2Vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/pretrained/w2v/model')

# load dataset
data = pd.read_csv('path/to/dataset.csv')

# define text cleaning function
def clean_text(text):
    # remove non-alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text)
    # convert to lowercase
    text = text.lower()
    # split into words
    words = text.split()
    # remove stopwords
    words = [word for word in words if word not in stopwords]
    # lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    # join words back into sentence
    cleaned_text = ' '.join(words)
    return cleaned_text

# clean text data
data['cleaned_text'] = data['text'].apply(clean_text)

# create Word2Vec embeddings for each post
X = np.zeros((data.shape[0], w2v_model.vector_size))
for i, post in enumerate(data['cleaned_text']):
    # split post into words
    words = post.split()
    # calculate average Word2Vec embedding for words in post
    post_vec = np.mean([w2v_model[word] for word in words if word in w2v_model.vocab], axis=0)
    X[i] = post_vec

# create target variable
y = np.where(data['label']=='mlm', 1, 0)

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# evaluate model on test set
score = lr_model.score(X_test, y_test)

# save trained model
import joblib
joblib.dump(lr_model, 'path/to/trained/model.joblib')

from flask import Flask, request, jsonify

# load trained model
lr_model = joblib.load('path/to/trained/model.joblib')

# create Flask app
app = Flask(__name__)

# define endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json(force=True)
    # clean text data
    cleaned_text = clean_text(data['text'])
    # create Word2Vec embedding for post
    post_vec = np.mean([w2v_model[word] for word in cleaned_text.split() if word in w2v_model.vocab], axis=0)
    # make prediction using trained model
    pred = lr_model.predict([post_vec])[0]
    # return prediction result as JSON
    if pred == 1:
        return jsonify({'prediction': 'mlm'})
    else:
        return jsonify({'prediction': 'not_mlm'})

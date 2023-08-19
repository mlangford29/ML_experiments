# train.py

import numpy as np
import pandas as pd
import gensim
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Set up preprocessing tools
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load pre-trained Word2Vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/pretrained/w2v/model')

# Load dataset
data = pd.read_csv('path/to/dataset.csv')

# Define text cleaning function
def clean_text(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

# Clean text data
data['cleaned_text'] = data['text'].apply(clean_text)

# Create Word2Vec embeddings for each post
X = np.zeros((data.shape[0], w2v_model.vector_size))
for i, post in enumerate(data['cleaned_text']):
    words = post.split()
    vectors = [w2v_model[word] for word in words if word in w2v_model.vocab]
    if vectors:
        post_vec = np.mean(vectors, axis=0)
        X[i] = post_vec

# Create target variable
y = np.where(data['label']=='mlm', 1, 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(lr_model, 'path/to/trained/model.joblib')
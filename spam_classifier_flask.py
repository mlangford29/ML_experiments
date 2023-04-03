# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Clean data
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Feature engineering
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# Train model
clf = MultinomialNB()
clf.fit(X, y)

# Deploy model using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    text = clean_text(text)
    X_test = vectorizer.transform([text])
    y_pred = clf.predict(X_test)
    return jsonify({'prediction': int(y_pred[0])})

if __name__ == '__main__':
    app.run(debug=True)

# Monitor model performance
y_pred_train = clf.predict(X)
accuracy_train = accuracy_score(y, y_pred_train)
print('Training accuracy:', accuracy_train)

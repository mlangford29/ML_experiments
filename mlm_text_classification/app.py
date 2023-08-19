# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import gensim
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

# Set up preprocessing tools
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load pre-trained Word2Vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/pretrained/w2v/model')

# Define text cleaning function
def clean_text(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    words = [lemmatizer.lemmatize(word) for word in words]
    cleaned_text = ' '.join(words)
    return cleaned_text

# Flask setup
app = Flask(__name__)
CORS(app)  # handle cross-origin requests

# Load trained model once
lr_model = joblib.load('path/to/trained/model.joblib')

# Define endpoint for making predictions
@app.route('/api/v1/predict_mlm', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        cleaned_text = clean_text(data['text'])
        words = cleaned_text.split()
        vectors = [w2v_model[word] for word in words if word in w2v_model.vocab]
        if vectors:
            post_vec = np.mean(vectors, axis=0)
            pred = lr_model.predict([post_vec])[0]
            return jsonify({'prediction': 'mlm' if pred == 1 else 'not_mlm'})
        else:
            return jsonify({'error': 'Unable to process the post'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
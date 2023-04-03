import pandas as pd
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Function to prepare data and train the model
def train_model():
    # Load the data
    data = pd.read_csv('song_lyrics_data.csv')

    # Load the pre-trained word embeddings
    model = api.load('word2vec-google-news-300')

    # Clean and preprocess the lyrics
    data['cleaned_lyrics'] = data['lyrics'].apply(lambda x: preprocess_text(x))

    # Generate word embeddings for the lyrics
    lyrics_embeddings = np.array([get_embedding(x, model) for x in data['cleaned_lyrics']])

    # Train the model
    model = Word2Vec(lyrics_embeddings, min_count=1)

    # Save the model
    model.save('song_recommendation_model.bin')

# Function to deploy the model using Flask
def deploy_model():
    # Load the model
    model = Word2Vec.load('song_recommendation_model.bin')

    # Initialize Flask
    app = Flask(__name__)

    # API endpoint for recommending songs based on input lyrics
    @app.route('/recommend', methods=['POST'])
    def recommend_songs():
        # Get the input lyrics
        input_lyrics = request.json['lyrics']

        # Preprocess the input lyrics
        cleaned_lyrics = preprocess_text(input_lyrics)

        # Generate word embeddings for the input lyrics
        input_embeddings = get_embedding(cleaned_lyrics, model)

        # Calculate cosine similarity between the input lyrics and all songs in the dataset
        similarity_scores = cosine_similarity(input_embeddings.reshape(1, -1), model.wv.vectors)

        # Get the top 10 recommended songs based on similarity scores
        recommended_song_indices = similarity_scores.argsort()[0][::-1][:10]
        recommended_songs = data.loc[recommended_song_indices, 'title'].tolist()

        # Return the recommended songs as a JSON response
        return jsonify({'recommended_songs': recommended_songs})

    # Run the Flask app
    app.run()

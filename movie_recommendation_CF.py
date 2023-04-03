import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from flask import Flask, request, jsonify

# Load movie ratings data
df = pd.read_csv('movie_ratings.csv')

# Clean the data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Feature engineering
# Convert movie titles to numerical IDs
df['movieId'] = df['movie_title'].astype('category').cat.codes
# Create a mapping from movie IDs to movie titles
id_to_title = dict(enumerate(df['movie_title'].astype('category').cat.categories))

# Train collaborative filtering model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Deploy the model using Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user ID and movie title from request
    user_id = int(request.form['user_id'])
    movie_title = request.form['movie_title']
    
    # Get movie ID from title
    movie_id = df.loc[df['movie_title'] == movie_title, 'movieId'].iloc[0]
    
    # Predict rating for movie and user
    prediction = algo.predict(user_id, movie_id)
    predicted_rating = prediction.est
    
    # Return prediction as JSON
    result = {'predicted_rating': predicted_rating, 'movie_title': movie_title}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

# import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
from flask import Flask, request, jsonify

# define function for data preparation and model training
def train_anomaly_detection_model(data_file_path):
    # read in data
    data = pd.read_csv(data_file_path)

    # split into training and test sets
    train_set = data.sample(frac=0.8, random_state=123)
    test_set = data.drop(train_set.index)

    # train isolation forest model
    clf = IsolationForest(contamination='auto', random_state=123)
    clf.fit(train_set.drop('timestamp', axis=1))

    # save model to disk
    with open('anomaly_detection_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

# define function for quantile binning engine
def anomaly_score_to_quantile(score):
    if score < -0.5:
        return 'low'
    elif score < 0:
        return 'medium'
    elif score < 0.5:
        return 'high'
    else:
        return 'very high'

# define flask app
app = Flask(__name__)

# load saved model
with open('anomaly_detection_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# define route for model prediction
@app.route('/predict', methods=['POST'])
def predict_anomaly():
    # get request data
    data = request.json

    # convert data to dataframe
    df = pd.DataFrame(data, index=[0])

    # make prediction
    score = clf.score_samples(df.drop('timestamp', axis=1))

    # convert anomaly score to quantile bin
    quantile_bin = anomaly_score_to_quantile(score)

    # return prediction
    return jsonify({'anomaly': bool(score < 0), 'quantile_bin': quantile_bin})

if __name__ == '__main__':
    # train and save model
    train_anomaly_detection_model('website_traffic_data.csv')

    # run flask app
    app.run(debug=True)

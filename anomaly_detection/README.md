# Anomaly Detection Flask API

This Flask application provides an API for training anomaly detection models and predicting anomalies on a live stream of data.

## Features
- **Model Training**: Accepts a dataset, trains an ensemble of anomaly detection models, and persists the model for future use.
- **Anomaly Prediction**: Takes in a live stream of data and uses the trained model to detect anomalies, returning results and severity scores.
- **Data Validation**: Ensures the integrity of incoming data with comprehensive validation checks.

## Setup and Installation

### Requirements
- Python 3.7+
- Flask
- pandas
- scikit-learn

## Usage

1. Start the Flask app:
   ```
   python app.py
   ```

2. **Model Training**:
   Make a POST request to `/train` with a dataset in JSON format to train the model. 

   Sample cURL command:
   ```
   curl --request POST 'http://127.0.0.1:5000/train' --header 'Content-Type: application/json' --data 'YOUR_JSON_DATA_HERE'
   ```

3. **Anomaly Prediction**:
   Make a POST request to `/predict` with data in JSON format to predict anomalies.

   Sample cURL command:
   ```
   curl --request POST 'http://127.0.0.1:5000/predict' --header 'Content-Type: application/json' --data 'YOUR_JSON_DATA_HERE'
   ```

## Error Handling

The API will return a relevant error message for invalid data or other issues.
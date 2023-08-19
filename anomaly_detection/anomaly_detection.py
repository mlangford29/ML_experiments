import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
import os

MODEL_PATH = 'anomaly_ensemble.pkl'

class AnomalyEnsemble:
    def __init__(self, contamination=0.1):
        """
        Initialize the ensemble with a set of common anomaly detection algorithms.
        :param contamination: The proportion of anomalies in the data.
        """
        self.models = [
            EllipticEnvelope(contamination=contamination),
            IsolationForest(contamination=contamination),
            OneClassSVM(nu=contamination)
        ]
    
    def fit(self, data):
        """
        Train each model on the data.
        :param data: The training dataset.
        """
        for model in self.models:
            model.fit(data)
    
    def predict(self, data):
        """
        Predict anomalies using majority voting from all models.
        :param data: The dataset to predict on.
        :return: List of predictions with 1 indicating normal and -1 indicating anomaly.
        """
        # Get predictions from all models
        predictions = [model.predict(data) for model in self.models]
        
        # Sum the predictions from all models for each data point
        summed = np.sum(predictions, axis=0)
        
        # Majority voting: Anomaly if at least 2 out of 3 models agree it's an anomaly
        ensemble_predictions = np.where(summed <= -1, -1, 1)
        
        return ensemble_predictions
    
    def decision_function(self, data):
        """
        Get averaged anomaly scores of the ensemble.
        :param data: The dataset to score.
        :return: Averaged anomaly scores.
        """
        # Get the anomaly scores from all models
        scores = [model.decision_function(data) for model in self.models]
        
        # Average the scores for ensemble
        return np.mean(scores, axis=0)

def train_model(data, contamination=0.1):
    """
    Train the ensemble on the given data.
    :param data: Training dataset.
    :param contamination: The proportion of anomalies in the data.
    :return: Success message.
    """
    ensemble = AnomalyEnsemble(contamination=contamination)
    ensemble.fit(data)
    
    # Persist the trained ensemble model to disk
    joblib.dump(ensemble, MODEL_PATH)
    
    return "Model trained successfully!"

def predict_anomalies(data):
    """
    Predict anomalies and their severity based on the trained ensemble.
    :param data: Dataset to predict anomalies on.
    :return: Indices of anomalies and their severity scores.
    """
    # Load the trained ensemble model from disk
    if os.path.exists(MODEL_PATH):
        ensemble = joblib.load(MODEL_PATH)
    else:
        raise Exception("Model not found. Train the model first.")
    
    # Get predictions from the ensemble
    preds = ensemble.predict(data)
    
    # Find the indices of anomalies (-1 represents anomalies)
    anomalies = np.where(preds == -1)
    
    # Calculate severity scores based on averaged anomaly scores from the ensemble
    scores = ensemble.decision_function(data)
    
    # Define quantile thresholds for severity
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    
    # Assign severity levels based on thresholds
    severity_scores = []
    for score in scores:
        if score <= q1:
            severity_scores.append("Low")
        elif score <= q3:
            severity_scores.append("Medium")
        else:
            severity_scores.append("High")
    
    return anomalies, severity_scores
from flask import Flask, request, jsonify
import pandas as pd
import anomaly_detection as ad

app = Flask(__name__)

def validate_input_data(df):
    """
    Validate the input data format.

    This function checks for the presence of NaN values and could be 
    expanded for other data validation checks.
    """
    # Check for NaN values
    if df.isnull().values.any():
        raise ValueError("Data contains NaN values.")

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.json
        df = pd.DataFrame(data)

        # Validate input data
        validate_input_data(df)

        message = ad.train_model(df)
        return jsonify({'message': message})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data)

        # Validate input data
        validate_input_data(df)

        anomalies, severity_scores = ad.predict_anomalies(df)
        return jsonify({'anomalies': anomalies.tolist(), 'severity_scores': severity_scores})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

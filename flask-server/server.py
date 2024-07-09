# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 


# Load the trained model
knn_model = joblib.load('knn_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Diabetes Prediction Model"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data['Age']
    gender = data['Gender']
    bmi = data['BMI']
    family_history = data['Family History of Diabetes']

    # Convert gender and family history to numerical values
    gender = 0 if gender == 'Female' else 1
    family_history = 0 if family_history == 'No' else 1

    # Create input feature array
    features = np.array([[age, gender, bmi, family_history]])
    
    # Standardize the features
    features = scaler.transform(features)

    # Make prediction
    prediction = knn_model.predict(features)

    # Convert prediction to readable format
    diagnosis = 'Yes' if prediction[0] == 1 else 'No'

    # Prepare the response
    response = {
        'Diagnosis': diagnosis
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

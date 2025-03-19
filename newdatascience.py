from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler from .pkl files
with open('svm2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler2.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index2.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the HTML form
        age = float(request.form['age'])
        systolic_bp = float(request.form['systolic_bp'])
        diastolic_bp = float(request.form['diastolic_bp'])
        cholesterol = float(request.form['cholesterol'])

        # Create a feature array and scale the data
        features = np.array([age, systolic_bp, diastolic_bp, cholesterol]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make the prediction using the model
        prediction = model.predict(features_scaled)

        # Return the prediction result as JSON
        return render_template('index2.html', prediction=prediction[0])

    except Exception as e:
        return render_template('index2.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

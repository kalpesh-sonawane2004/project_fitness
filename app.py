from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("bmi_classifier.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("Bmi.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        age = data['Age']
        height = data['Height']
        weight = data['Weight']
        
        prediction = model.predict([[age, height, weight]])[0]
        result = "Fit" if prediction == 1 else "Not Fit"
        
        return jsonify({"Fitness Status": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

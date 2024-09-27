from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained logistic regression model and scaler
logit_model = joblib.load(r'C:\Users\Dell\Desktop\MSC PROJECT\CVDAPP\smote_logistic_regression_model_v1.pkl')
scaler = joblib.load(r'C:\Users\Dell\Desktop\MSC PROJECT\CVDAPP\scaler.pkl')

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html', prediction=None, probability=None, note=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form and create a DataFrame
    data = {
        'SEX': int(request.form['SEX']),
        'TOTCHOL': int(request.form['TOTCHOL']),
        'AGE': int(request.form['AGE']),
        'CURSMOKE': int(request.form['CURSMOKE']),
        'BMI': int(request.form['BMI']),
        'DIABETES': int(request.form['DIABETES']),
        'BPMEDS': int(request.form['BPMEDS']),
        'HEARTRTE': int(request.form['HEARTRTE']),
        'GLUCOSE': int(request.form['GLUCOSE']),
        'HYPERTEN': int(request.form['HYPERTEN']),
        'PREVIOUS_CVD_EVENT': int(request.form['PREVIOUS_CVD_EVENT']),
        'BPMAP': int(request.form['BPMAP'])
    }

    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction_proba = logit_model.predict_proba(input_data_scaled)[:, 1]
    risk_percentage = prediction_proba[0] * 100

    # Determine feedback note based on the risk percentage
    if risk_percentage <= 20:
        note = f"Low Risk ({risk_percentage:.2f}%). You are currently at a low risk of developing cardiovascular disease (CVD). Keep up the good work by maintaining a healthy lifestyle that includes regular physical activity, a balanced diet, and regular health check-ups. Continue monitoring key health metrics such as blood pressure, cholesterol levels, and blood sugar to ensure you stay on track."
    elif 21 <= risk_percentage <= 40:
        note = f"Moderate Risk ({risk_percentage:.2f}%). You have a moderate risk of developing cardiovascular disease (CVD). This is a good time to consider lifestyle changes that can further reduce your risk, such as increasing your physical activity, improving your diet, and managing your stress levels."
    elif 41 <= risk_percentage <= 60:
        note = f"Elevated Risk ({risk_percentage:.2f}%). You are at an elevated risk of cardiovascular disease (CVD). It is important to take action now to lower your risk. Consider working with a healthcare provider to develop a personalized plan that includes regular exercise, a heart-healthy diet, and potentially medication to control blood pressure, cholesterol, or diabetes. Early intervention can greatly reduce your long-term risk."
    elif 61 <= risk_percentage <= 80:
        note = f"High Risk ({risk_percentage:.2f}%). You are at a high risk of developing cardiovascular disease (CVD). We strongly advise you to consult with a healthcare provider as soon as possible to discuss your risk factors and potential interventions."
    else:
        note = f"Very High Risk ({risk_percentage:.2f}%). You are at a very high risk of cardiovascular disease (CVD). Immediate medical attention is required. You may need lifestyle changes, medication, and regular monitoring to mitigate the risk of serious cardiovascular events such as a heart attack or stroke."

    # Render the home page with the prediction result
    return render_template('index.html', prediction=int(prediction_proba[0] >= 0.5), probability=risk_percentage, note=note)

if __name__ == '__main__':
    app.run(debug=True, port=5004)

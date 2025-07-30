from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000
data = {
    'Age': np.random.randint(30, 80, n_samples),
    'Smoking': np.random.randint(0, 2, n_samples),
    'Alcohol': np.random.randint(0, 2, n_samples),
    'Exercise': np.random.randint(0, 2, n_samples),
    'Cholesterol': np.random.randint(150, 300, n_samples),
    'BloodPressure': np.random.randint(100, 220, n_samples)
}
df = pd.DataFrame(data)

# Define Risk
df['Risk'] = (
    (df['Age'] > 55).astype(int) +
    (df['Smoking'] == 1).astype(int) +
    (df['Alcohol'] == 1).astype(int) +
    (df['Exercise'] == 0).astype(int) +
    (df['Cholesterol'] > 220).astype(int) +
    (df['BloodPressure'] > 140).astype(int)
)
df['Risk'] = (df['Risk'] >= 3).astype(int)

# Train model
X = df[['Age', 'Smoking', 'Alcohol', 'Exercise', 'Cholesterol', 'BloodPressure']]
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    smoking = int(request.form['smoking'])
    alcohol = int(request.form['alcohol'])
    exercise = int(request.form['exercise'])
    cholesterol = int(request.form['cholesterol'])
    blood_pressure = int(request.form['blood_pressure'])

    user_data = pd.DataFrame([[age, smoking, alcohol, exercise, cholesterol, blood_pressure]],
                             columns=X.columns)

    risk_prediction = model.predict(user_data)[0]
    risk_probability = model.predict_proba(user_data)[0][1]

    if risk_prediction == 1:
        message = "⚠ HIGH RISK of stroke or heart attack."
        tips = {
            "lifestyle": [
                "Quit smoking and limit alcohol consumption.",
                "Get at least 7-8 hours of sleep per night.",
                "Monitor blood pressure and cholesterol regularly."
            ],
            "exercise": [
                "Aim for 30 minutes of moderate activity daily (e.g., walking, cycling).",
                "Include resistance training 2–3 times per week."
            ],
            "diet": [
                "Reduce salt and saturated fat intake.",
                "Eat more fruits, vegetables, and whole grains.",
                "Limit red and processed meats."
            ]
        }
    else:
        message = "✅ LOW RISK of stroke or heart attack."
        tips = {
            "lifestyle": [
                "Maintain healthy habits and routine checkups.",
                "Stay hydrated and manage stress effectively."
            ],
            "exercise": [
                "Continue regular physical activity.",
                "Try yoga or stretching to support cardiovascular health."
            ],
            "diet": [
                "Stick to a balanced diet rich in nutrients.",
                "Include omega-3 rich foods like fish and flaxseed."
            ]
        }

    return render_template('result.html', prob=round(risk_probability, 2), msg=message, tips=tips)

if __name__ == '__main__':
    app.run(debug=True,port=5500)
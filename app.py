import random
from flask import Flask, render_template, request
from sklearn.metrics import classification_report
import numpy as np
import joblib  

app = Flask(__name__)

# Load model and metrics
model_bundle = joblib.load("lr_best_model.pkl")
model = model_bundle["model"]
metrics = model_bundle["metrics"]
predictions = model_bundle["predictions"]   


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = int(request.form["gender"])
    age = int(request.form["age"])
    sleep_duration = float(request.form["sleep_duration"])
    sleep_quality = float(request.form["sleep_quality"])
    physical_activity = float(request.form["physical_activity"])
    stress_level = float(request.form["stress_level"])
    bmi_category = int(request.form["bmi_category"])
    heart_rate = float(request.form["heart_rate"])
    daily_steps = float(request.form["daily_steps"])

    # Arrange inputs in correct ORDER
    features = np.array([[gender, age, sleep_duration, sleep_quality,
                          physical_activity, stress_level, bmi_category,
                          heart_rate, daily_steps]])

    # Predict using model
    pred_num = model.predict(features)[0]
    accuracy = round(metrics["accuracy"] * 100, 2)


     # Suggestions for each category (5 examples each)
    suggestions_dict = {
        "Normal": [
            "Your sleep patterns are healthy. Continue maintaining a balanced lifestyle with regular exercise and good nutrition.",
            "Great job! Ensure you maintain consistent sleep schedules and manage stress effectively to keep your sleep quality high.",
            "You have a normal sleep pattern. Keep monitoring your sleep duration and quality to prevent future disruptions."
        ],
        "Sleep Apnea": [
            "Signs of sleep apnea detected. Consider consulting a sleep specialist for proper evaluation and treatment options.",
            "You may be at risk for sleep apnea. Improving sleep hygiene and avoiding alcohol before bed can help, but a medical check-up is recommended.",
            "Your results suggest potential sleep apnea. Using a CPAP device or positional therapy may help after consultation with a healthcare professional."
        ],
        "Insomnia": [
            "Insomnia risk detected. Establish a consistent sleep schedule and avoid screens at least one hour before bed.",
            "Your results suggest difficulty sleeping. Relaxation techniques like meditation or deep breathing may improve your sleep quality.",
            "High insomnia risk observed. Limit caffeine intake and maintain a calm bedtime routine to enhance sleep duration and quality."
        ]
    }

    # Map prediction to text and color
    if pred_num == 1:
        prediction = "Normal"
        color = "green"
        severity = 10
    elif pred_num == 2:
        prediction = "Sleep Apnea"
        color = "orange"
        severity = 60
    else:
        prediction = "Insomnia"
        color = "red"
        severity = 80

    # Random suggestion from the set for the prediction
    suggestion = random.choice(suggestions_dict[prediction])

    return render_template("index.html", 
                           prediction=prediction,
                           accuracy=accuracy,
                           color=color,
                           suggestion=suggestion,
                           severity=severity,
                           predictions=predictions
                           )
if __name__ == "__main__":
   # app.run(debug=True, port=5001)
     app.run(debug=True)

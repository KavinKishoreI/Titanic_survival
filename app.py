from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import traceback
import warnings

# Suppress scikit-learn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

app = Flask(__name__)
app.secret_key = "titanic_secret"  # Needed for session storage

# Load model with error handling
try:
    model = joblib.load("titanic_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to retrain the model...")
    
    # If model loading fails, try to retrain it
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        
        # Load and prepare data
        import seaborn as sns
        df = sns.load_dataset("titanic")
        
        # Clean data (same as in titanic_ds.py)
        df = df.drop(columns=['who', 'alive', 'adult_male', 'deck', 'alone', 'embark_town', 'class'])
        df['relatives'] = df['sibsp'] + df['parch']
        df = df.drop(columns=['sibsp', 'parch'])
        df['age'].fillna(df['age'].median(), inplace=True)
        df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
        df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Prepare features and target
        X = df.drop("survived", axis=1)
        y = df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)
        
        # Train new model
        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        
        # Save the new model
        joblib.dump(model, "titanic_model.pkl")
        print("New model trained and saved successfully!")
        
    except Exception as retrain_error:
        print(f"Failed to retrain model: {retrain_error}")
        model = None

# Questions for Q&A style (keeping for backward compatibility)
questions = [
    {"name": "pclass", "text": "Which class did you travel in? (1 = First, 2 = Second, 3 = Third)"},
    {"name": "sex", "text": "What is your gender? (0 = Male, 1 = Female)"},
    {"name": "age", "text": "How old are you?"},
    {"name": "fare", "text": "What was your ticket fare?"},
    {"name": "embarked", "text": "Where did you embark? (0 = Southampton, 1 = Cherbourg, 2 = Queenstown)"},
    {"name": "relatives", "text": "How many relatives were with you on board?"}
]

@app.route("/")
def home():
    session["answers"] = {}  # reset answers
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """New endpoint for the single-page app"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ["pclass", "sex", "age", "fare", "embarked", "relatives"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
        
        # Validate and convert data types
        try:
            pclass = int(data["pclass"])
            sex = int(data["sex"])
            age = float(data["age"])
            fare = float(data["fare"])
            embarked = int(data["embarked"])
            relatives = int(data["relatives"])
        except ValueError:
            return jsonify({"error": "Invalid data types"}), 400
        
        # Validate ranges
        if not (1 <= pclass <= 3):
            return jsonify({"error": "Passenger class must be 1, 2, or 3"}), 400
        if not (0 <= sex <= 1):
            return jsonify({"error": "Sex must be 0 (Male) or 1 (Female)"}), 400
        if age < 0 or age > 120:
            return jsonify({"error": "Age must be between 0 and 120"}), 400
        if fare < 0:
            return jsonify({"error": "Fare cannot be negative"}), 400
        if not (0 <= embarked <= 2):
            return jsonify({"error": "Embarked must be 0, 1, or 2"}), 400
        if relatives < 0:
            return jsonify({"error": "Relatives cannot be negative"}), 400
        
        # Prepare input for model (same order as training)
        features = [[pclass, sex, age, fare, embarked, relatives]]
        
        # Make prediction
        prob = model.predict_proba(features)[0][1] * 100  # probability of survival
        
        return jsonify({"probability": round(prob, 2)})
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    """Legacy route for template-based approach"""
    if request.method == "POST":
        # Save answer
        answer = request.form["answer"]
        if qid > 0 and qid <= len(questions):  # Fix indexing bug
            session["answers"][questions[qid - 1]["name"]] = float(answer)

    if qid >= len(questions):
        return redirect(url_for("result"))

    return render_template("question.html", question=questions[qid], qid=qid)

@app.route("/result")
def result():
    """Legacy route for template-based approach"""
    answers = session.get("answers", {})
    if len(answers) < len(questions):
        return redirect(url_for("home"))

    try:
        # Prepare input for model (same order as training)
        features = [[
            answers["pclass"],
            answers["sex"],
            answers["age"],
            answers["fare"],
            answers["embarked"],
            answers["relatives"]
        ]]
        
        prob = model.predict_proba(features)[0][1] * 100  # probability of survival
        return render_template("result.html", prob=round(prob, 2))
    except Exception as e:
        print(f"Result error: {e}")
        return render_template("result.html", prob=0, error="Prediction failed")

if __name__ == "__main__":
    app.run(debug=True)

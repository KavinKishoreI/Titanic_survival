# 🚢 Titanic Survival Predictor

A machine learning web application that predicts the survival probability of passengers on the Titanic based on their characteristics. Built with Flask, scikit-learn, and modern web technologies.

## ✨ Features
- **Interactive Q&A Interface**: Multiple-choice and number input questions
- **Real-time Predictions**: Instant survival probability calculations
- **Beautiful UI**: Modern design with Tailwind CSS
- **Input Validation**: Comprehensive validation for all inputs
- **Progress Tracking**: Visual progress bar
- **Error Handling**: Robust error handling and feedback

## 🏗️ Project Structure

```
titanic-survival/
├── app.py                 # Flask web application
├── titanic_ds.py         # Data science script for model training
├── titanic_model.pkl     # Trained machine learning model
├── requirements.txt       # Python dependencies
├── static/
│   └── style.css         # Additional CSS styles
├── templates/
│   └── index.html        # Main application interface
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser

### Installation & Running

1. **Install dependencies**
   pip install -r requirements.txt

2. **Run the application**
   python app.py

3. **Open browser**
   Navigate to: http://127.0.0.1:5000

## 📋 How to Use

1. **Passenger Class**: Choose First 🥇, Second 🥈, or Third 🥉 class
2. **Gender**: Select Male 👨 or Female 👩
3. **Age**: Enter your age (0-120 years)
4. **Fare**: Enter ticket price in pounds
5. **Embarkation**: Choose Southampton 🇬🇧, Cherbourg 🇫🇷, or Queenstown 🇮🇪
6. **Relatives**: Count family members traveling with you

## 🔧 Technical Details

- **Algorithm**: Logistic Regression
- **Backend**: Flask (Python)
- **Frontend**: HTML5, JavaScript, Tailwind CSS
- **ML Libraries**: scikit-learn, pandas, numpy
- **Model**: Automatically retrains if loading fails

---

**Happy Predicting! 🚢✨**

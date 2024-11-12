from flask import Flask, render_template, request, jsonify
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def clean_text(text):
    return re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters

def explain_prediction(text, prediction):
    explanation = "The model detected patterns in word usage and structure commonly associated with "
    if prediction == 1:
        explanation += "fake news. Factors like exaggerated language, sensational phrasing, or unusual word pairings might have contributed."
    else:
        explanation += "real news. Indicators such as formal language and verifiable claims contributed to this assessment."
    return explanation

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    explanation = None

    if request.method == "POST":
        # Get user input
        user_text = request.form["news_text"]

        # Preprocess and make prediction
        processed_text = clean_text(user_text)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)[0]

        # Explanation for the prediction
        explanation = explain_prediction(user_text, prediction)

    return render_template("index.html", prediction=prediction, explanation=explanation)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess and make prediction
    processed_text = clean_text(user_text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)[0]

    explanation = explain_prediction(user_text, prediction)

    return jsonify({
        "prediction": "Fake News" if prediction == 1 else "Real News",
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)

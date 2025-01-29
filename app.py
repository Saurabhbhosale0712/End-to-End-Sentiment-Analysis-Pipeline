# from flask import Flask, request, jsonify
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# import requests

# # Initialize Flask app
# app = Flask(__name__)

# # Load the pre-trained model
# model_path = "model.pkl"  # If it's in the same directory as app.py
# vectorizer_path = "tfidf_vectorizer.pkl"  # Update with your vectorizer file if provided

# try:
#     with open(model_path, "rb") as model_file:
#         model = pickle.load(model_file)
# except FileNotFoundError:
#     raise Exception("Model file not found. Please ensure the model.pkl file is available.")

# # Load the vectorizer if available
# try:
#     with open(vectorizer_path, "rb") as vectorizer_file:
#         vectorizer = pickle.load(vectorizer_file)
# except FileNotFoundError:
#     vectorizer = None  # Handle cases where no vectorizer is provided


# @app.route('/')
# def home():
#     return "Welcome to the Sentiment Prediction API! Use the /predict endpoint to get sentiment predictions."


# @app.route('/predict', methods=['POST'])
# def predict_sentiment():
#     # Parse input JSON
#     data = request.get_json()
#     review_text = data.get('review_text', "")

#     if not review_text:
#         return jsonify({"error": "Invalid input. 'review_text' field is required."}), 400

#     # Preprocess input (ensure vectorizer is used if provided)
#     if vectorizer:
#         transformed_text = vectorizer.transform([review_text])
#     else:
#         return jsonify({"error": "Vectorizer not available for preprocessing."}), 500

#     # Make prediction
#     prediction = model.predict(transformed_text)
#     sentiment = "positive" if prediction[0] == 1 else "negative"

#     # Return prediction as JSON
#     return jsonify({"sentiment_prediction": sentiment})


# # Test the Flask app locally with a POST request
# def test_local():
#     url = "http://127.0.0.1:5000/predict"
#     data = {"review_text": "This movie was fantastic!"}
#     response = requests.post(url, json=data)
#     print("Response:", response.json())


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import pickle
# import re

# # Initialize Flask app
# app = Flask(__name__)

# # Load the pre-trained model and vectorizer
# model_path = "model.pkl"  # Path to your trained model
# vectorizer_path = "tfidf_vectorizer.pkl"  # Path to your TF-IDF vectorizer

# # Load the model
# try:
#     with open(model_path, "rb") as model_file:
#         model = pickle.load(model_file)
# except FileNotFoundError:
#     raise Exception("Model file not found. Please ensure the model.pkl file is available.")

# # Load the vectorizer
# try:
#     with open(vectorizer_path, "rb") as vectorizer_file:
#         vectorizer = pickle.load(vectorizer_file)
# except FileNotFoundError:
#     raise Exception("Vectorizer file not found. Please ensure the tfidf_vectorizer.pkl file is available.")

# # Define a simple text preprocessing function
# def preprocess_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
#     text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#     text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
#     text = text.strip()  # Strip leading and trailing whitespace
#     return text

# @app.route('/')
# def home():
#     return "Welcome to the Sentiment Prediction API! Use the /predict endpoint to get sentiment predictions."

# @app.route('/predict', methods=['POST'])
# def predict_sentiment():
#     # Parse input JSON
#     data = request.get_json()
#     review_text = data.get('review_text', "")

#     if not review_text:
#         return jsonify({"error": "Invalid input. 'review_text' field is required."}), 400

#     # Preprocess input
#     cleaned_text = preprocess_text(review_text)

#     # Transform the cleaned text to TF-IDF vector
#     transformed_text = vectorizer.transform([cleaned_text])

#     # Make prediction
#     prediction = model.predict(transformed_text)
#     sentiment = "positive" if prediction[0] == 1 else "negative"

#     # Return prediction as JSON
#     return jsonify({"sentiment_prediction": sentiment})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template
import pickle
import re

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Function to preprocess the text (same as in your previous code)
def preprocess_text_simple(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers (retain only alphabets and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling review submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_review = request.form['review']
        cleaned_review = preprocess_text_simple(input_review)
        review_tfidf = tfidf_vectorizer.transform([cleaned_review])
        predicted_sentiment = model.predict(review_tfidf)[0]

        # Display the original, cleaned, and predicted sentiment
        return render_template('result.html', 
                               input_review=input_review, 
                               cleaned_review=cleaned_review, 
                               predicted_sentiment=predicted_sentiment)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load or train emotion recognition model
def detect_emotion(text):
    # Example: Load pre-trained model
    try:
        vectorizer = joblib.load("models/vectorizer.pkl")
        model = joblib.load("models/emotion_model.pkl")
    except:
        # Train a simple model if not found
        emotions = ["happy", "sad", "angry", "neutral"]
        texts = ["I feel great!", "I am so sad.", "This makes me angry.", "I am fine."]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        model = SVC()
        model.fit(X, emotions)
        joblib.dump(vectorizer, "models/vectorizer.pkl")
        joblib.dump(model, "models/emotion_model.pkl")

    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
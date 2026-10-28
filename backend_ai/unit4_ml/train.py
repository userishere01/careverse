import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_symptom_model():
    data_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset missing at {data_path}")

    df = pd.read_csv(data_path)
    X = df["text"]
    y = df["specialist"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=6, random_state=42, stratify=y
    )

    # Unit IV Pipeline: TF-IDF + Multinomial Naive Bayes (Laplace smoothed)
    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    print("Training CareVerse Unit-IV Naive Bayes Model...")
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\nModel Trained Successfully!")
    print(f"Validation Accuracy: {acc * 100:.2f}%\n")
    print("Classification Summary:\n", classification_report(y_test, y_pred, zero_division=0))

    save_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model_pipeline, save_path)
    print(f"Model saved successfully at: {save_path}")

if __name__ == "__main__":
    train_symptom_model()
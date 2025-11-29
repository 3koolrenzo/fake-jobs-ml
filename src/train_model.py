import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ================================
# File Paths
# ================================
DATA_PATH = "data/fake_job_postings.csv"
MODEL_PATH = "models/fakejob_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"
CONF_MATRIX_PATH = "reports/confusion_matrix.png"


# ================================
# Load + Clean Dataset
# ================================
def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads the Fake Job Posting dataset and prepares text + label columns.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # "fraudulent" column: 1 = fake job, 0 = real job
    if "fraudulent" not in df.columns:
        raise ValueError("Expected column 'fraudulent' not found.")

    # Combine multiple text fields into one
    text_columns = [
        "title",
        "location",
        "department",
        "company_profile",
        "description",
        "requirements",
        "benefits",
    ]

    df["text"] = df[text_columns].astype(str).apply(lambda row: " ".join(row), axis=1)

    df = df[["text", "fraudulent"]]
    df = df.rename(columns={"fraudulent": "label"})

    # Remove missing
    df = df.dropna(subset=["text"])

    return df


# ================================
# Training Function
# ================================
def train():
    print("[1] Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X = df["text"]
    y = df["label"]

    print("\n[2] Splitting train/test set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3] Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("[4] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    print("[5] Evaluating model...")
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Job", "Fake Job"]))

    print("\n[6] Saving confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real Job", "Fake Job"],
        yticklabels=["Real Job", "Fake Job"]
    )
    plt.xlabel("Predicted Jobs")
    plt.ylabel("Actual Jobs")
    plt.title("Fake Job Posting Detection Confusion Matrix")
    plt.tight_layout()

    os.makedirs(os.path.dirname(CONF_MATRIX_PATH), exist_ok=True)
    plt.savefig(CONF_MATRIX_PATH)
    plt.close()

    print(f"Confusion matrix saved: {CONF_MATRIX_PATH}")

    print("\n[7] Saving model + vectorizer...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")

    print("\nDone!")


# ================================
# Run
# ================================
if __name__ == "__main__":
    train()

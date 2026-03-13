from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from features_message import extract_manual_features


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "fraud_messages.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


class ManualFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame([extract_manual_features(text) for text in X])


def load_data(path: Path):

    df = pd.read_csv(path)

    if "message" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'message' and 'label' columns")

    return df.dropna(subset=["message", "label"])


def main():

    df = load_data(DATA_PATH)

    X = df["message"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=3000
                ),
                "message"
            ),

            (
                "manual",
                Pipeline(
                    steps=[
                        ("extract", ManualFeatureTransformer()),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0))
                    ]
                ),
                "message"
            ),

        ]
    )

    model = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=5,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    model.fit(X_train.to_frame(), y_train)

    preds = model.predict(X_test.to_frame())
    probs = model.predict_proba(X_test.to_frame())[:, 1]

    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("Classification Report")
    print(report)

    print("\nROC AUC:", round(auc, 4))

    print("\nConfusion Matrix")
    print(cm)

    joblib.dump(model, MODEL_DIR / "message_fraud_model.pkl")

    metrics = {
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }

    with open(MODEL_DIR / "message_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel saved to:", MODEL_DIR / "message_fraud_model.pkl")
    print("Metrics saved to:", MODEL_DIR / "message_model_metrics.json")


if __name__ == "__main__":
    main()

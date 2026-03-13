import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


df = pd.read_csv(DATA_PATH)

X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    ))

])


pipeline.fit(X_train, y_train)


preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:,1]


print("\nClassification Report\n")
print(classification_report(y_test, preds))

auc = roc_auc_score(y_test, probs)

print("\nROC AUC:", auc)


joblib.dump(pipeline, MODEL_DIR / "transaction_fraud_model.pkl")

print("\nTransaction model saved.")

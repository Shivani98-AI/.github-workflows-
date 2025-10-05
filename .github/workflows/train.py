# tourism_project/train.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib, os
from huggingface_hub import HfApi, upload_file
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_REPO = "Shivani1223/tourism_best_model"

# Load prepared splits
train_df = pd.read_csv("tourism_project/prepared/train.csv")
test_df = pd.read_csv("tourism_project/prepared/test.csv")

TARGET = "ProdTaken"
X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

# Encode categoricals
cat_cols = X_train.select_dtypes(include="object").columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

# Candidate models
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                             random_state=42, eval_metric="logloss")
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {"accuracy": acc, "f1": f1}
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}")

# Pick best
best_model_name = max(results, key=lambda k: results[k]["f1"])
best_model = models[best_model_name]

# Save locally
os.makedirs("tourism_project/model_building", exist_ok=True)
model_path = f"tourism_project/model_building/{best_model_name}.pkl"
joblib.dump(best_model, model_path)

# Upload to Hugging Face Hub
api = HfApi()
api.create_repo(repo_id=MODEL_REPO, token=HF_TOKEN, repo_type="model", exist_ok=True)

upload_file(path_or_fileobj=model_path, path_in_repo=f"{best_model_name}.pkl",
            repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN)

print(f"✅ Best model ({best_model_name}) uploaded to https://huggingface.co/{MODEL_REPO}")

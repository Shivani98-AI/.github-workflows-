# tourism_project/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, upload_file
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATASET_REPO = "Shivani1223/tourism_dataset"

# Load raw dataset (assuming you uploaded tourism.csv earlier)
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop unnecessary columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Handle missing values
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Split
TARGET = "ProdTaken"
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)

# Save locally
train_path = "tourism_project/prepared/train.csv"
test_path = "tourism_project/prepared/test.csv"
os.makedirs("tourism_project/prepared", exist_ok=True)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Upload to Hugging Face Hub
api = HfApi()
api.create_repo(repo_id=DATASET_REPO, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

upload_file(path_or_fileobj=train_path, path_in_repo="prepared/train.csv",
            repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
upload_file(path_or_fileobj=test_path, path_in_repo="prepared/test.csv",
            repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)

print("✅ Data preparation complete and uploaded.")

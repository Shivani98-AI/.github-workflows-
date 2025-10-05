from huggingface_hub import HfApi, upload_file
import os

HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_REPO = "Shivani1223/tourism_dataset"

api = HfApi()
api.create_repo(repo_id=DATASET_REPO, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

upload_file(
    path_or_fileobj=".github/workflows/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO,
    repo_type="dataset",
    token=HF_TOKEN
)
print("Dataset uploaded")

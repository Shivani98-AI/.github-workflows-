from huggingface_hub import upload_file
import os

HF_TOKEN = os.environ["HF_TOKEN"]
SPACE_REPO = "Shivani1223/tourism_app"

files = {
    "tourism_project/deployment/app.py": "app.py",
    "tourism_project/deployment/requirements.txt": "requirements.txt",
    "tourism_project/deployment/inference_utils.py": "inference_utils.py"
}

for src, dest in files.items():
    upload_file(
        path_or_fileobj=src,
        path_in_repo=dest,
        repo_id=SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN
    )
print("✅ Space updated")

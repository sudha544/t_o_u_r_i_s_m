!pip install datasets

from huggingface_hub import HfApi
import os

# Replace with your Hugging Face username and the desired dataset repository name
HF_USERNAME = "sudha1726"
HF_DATASET_REPO = "tourism-package-purchase-dataset"

# Get your Hugging Face token from Colab secrets
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')

# Initialize the HfApi
api = HfApi()

# Create the dataset repository on Hugging Face Hub
api.create_repo(
    repo_id=f"{HF_USERNAME}/{HF_DATASET_REPO}",
    repo_type="dataset",
    exist_ok=True,
    token=HF_TOKEN
)

# Define the path to the dataset file
dataset_file_path = "/content/tourism_project/data/tourism.csv"

# Upload the dataset file to the repository
api.upload_file(
    path_or_fileobj=dataset_file_path,
    path_in_repo="tourism.csv",
    repo_id=f"{HF_USERNAME}/{HF_DATASET_REPO}",
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"Dataset uploaded to https://huggingface.co/datasets/{HF_USERNAME}/{HF_DATASET_REPO}")

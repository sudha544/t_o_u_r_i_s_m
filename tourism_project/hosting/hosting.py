from huggingface_hub import HfApi, create_repo, upload_folder
from google.colab import userdata
import os

# Get your Hugging Face token from Colab secrets
HF_TOKEN = userdata.get('HF_TOKEN')

# Replace with your Hugging Face username and desired Space repository name
HF_USERNAME = "sudha1726"
HF_SPACE_REPO = "tourism-package-purchase-app" # Replace with your desired Hugging Face Space repo name

# Initialize the HfApi
api = HfApi()

# Create the repository on Hugging Face Hub if it doesn't exist
create_repo(repo_id=f"{HF_USERNAME}/{HF_SPACE_REPO}", repo_type="space", exist_ok=True, token=HF_TOKEN, space_sdk="docker")

# Define the local path to the deployment folder
deployment_folder_path = "tourism_project/deployment"

# Upload the entire deployment folder to the Hugging Face Space repository
upload_folder(
    folder_path=deployment_folder_path,
    repo_id=f"{HF_USERNAME}/{HF_SPACE_REPO}",
    repo_type="space",
    token=HF_TOKEN,
)

print(f"Deployment files uploaded to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{HF_SPACE_REPO}")

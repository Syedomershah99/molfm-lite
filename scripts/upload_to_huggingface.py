#!/usr/bin/env python
"""Upload MolFM-Lite models to Hugging Face Hub"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Configuration
HF_USERNAME = "OmerShah"
REPO_NAME = "molfm-lite"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

def main():
    # Initialize API
    api = HfApi()

    # Check if logged in
    try:
        whoami = api.whoami()
        print(f"Logged in as: {whoami['name']}")
    except Exception as e:
        print("Not logged in. Please run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        return

    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"Repository created/exists: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Upload folder
    upload_dir = Path(__file__).parent.parent / "huggingface_upload"

    if not upload_dir.exists():
        print(f"Upload directory not found: {upload_dir}")
        return

    print(f"\nUploading files from: {upload_dir}")
    print("Files to upload:")
    for f in upload_dir.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    # Upload
    try:
        upload_folder(
            folder_path=str(upload_dir),
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload MolFM-Lite model checkpoints and config"
        )
        print(f"\nUpload complete!")
        print(f"View your model at: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Error uploading: {e}")


if __name__ == "__main__":
    main()

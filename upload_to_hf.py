from huggingface_hub import HfApi
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the folder containing your merged model (config.json, model.safetensors, etc.)
MODEL_FOLDER = "./path/to/your/merged_model" 
# Your Hugging Face repo ID (e.g., "username/qwen-merged-lora")
REPO_ID = "your-hf-username/your-repo-name"

def upload_model():
    if not os.path.exists(MODEL_FOLDER):
        print(f"Error: The folder '{MODEL_FOLDER}' does not exist.")
        print("Please update MODEL_FOLDER with the path to your merged model.")
        return
        
    api = HfApi()
    
    print(f"Uploading {MODEL_FOLDER} to https://huggingface.co/{REPO_ID}...")
    
    # Create the repo if it doesn't exist. Set to public.
    api.create_repo(repo_id=REPO_ID, exist_ok=True, private=False)
    
    # Upload the folder
    api.upload_folder(
        folder_path=MODEL_FOLDER,
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_model()

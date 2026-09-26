import os
from huggingface_hub import hf_hub_download

MODELS_TO_DOWNLOAD = [
    # repo_id, filename, subfolder/local_dir
    ("Kim2091/UltraSharp", "4x-UltraSharp.safetensors", "models/UltraSharp"),
    ("OzzyGT/DAT_X2", "DAT_x2.safetensors", "models/DAT"),
    ("OzzyGT/DAT_X3", "DAT_x3.safetensors", "models/DAT"),
    ("OzzyGT/DAT_X4", "DAT_x4.safetensors", "models/DAT"),
    ("OzzyGT/4xNomosWebPhoto_RealPLKSR", "4xNomosWebPhoto_RealPLKSR.safetensors", "models/RealPLKSR"),
    ("Phips/4xRealWebPhoto_v4_dat2", "4xRealWebPhoto_v4_dat2.safetensors", "models/RealWebPhoto"),
    ("OzzyGT/4xRemacri", "4x_foolhardy_Remacri.safetensors", "models/4xRemacri"),
    ("OzzyGT/RealESRGAN_general_x4v3", "general_x4v3.safetensors", "models/RealESRGAN"),
    ("OzzyGT/RealESRGAN_anime_v3", "anime_x4v3.safetensors", "models/RealESRGAN"),
]

def download_all_models():
    for repo_id, filename, local_dir in MODELS_TO_DOWNLOAD:
        print(f"Downloading {filename} from {repo_id} to {local_dir}...")
        os.makedirs(local_dir, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename} from {repo_id}: {e}")

if __name__ == "__main__":
    download_all_models()

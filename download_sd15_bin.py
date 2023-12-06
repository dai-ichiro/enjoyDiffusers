import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

repo_id="runwayml/stable-diffusion-v1-5"
folder = os.path.basename(repo_id)

hf_hub_download(
    repo_id=repo_id, 
    filename="model_index.json",
    local_dir=folder,
    local_dir_use_symlinks=False	
)

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[
        #feature_extractor
        "feature_extractor/*",
        #safety_checker
        "safety_checker/config.json",
        "safety_checker/pytorch_model.bin",
        #scheduler
        "scheduler/*",
        #text_encoder
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        #tokenizer
        "tokenizer/*",
        #unet
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        #vae
        "vae/config.json",
        "vae/diffusion_pytorch_model.bin"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)
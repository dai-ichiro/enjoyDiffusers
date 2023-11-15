import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

repo_id="stabilityai/stable-diffusion-xl-base-1.0"
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
        #scheduler
        "scheduler/*",
        #text_encoder
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        #text_encoder_2
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors",
        #tokenizer
        "tokenizer/*",
        #tokenizer_2
        "tokenizer_2/*",
        #unet
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        #vae
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors"
    ],
    local_dir=folder,
    local_dir_use_symlinks=False
)


import os
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

repo_id="lllyasviel/control_v11p_sd15_openpose"
folder = os.path.basename(repo_id)

hf_hub_download(
    repo_id=repo_id, 
    filename="diffusion_pytorch_model.safetensors",
    local_dir=folder,
    local_dir_use_symlinks=False	
)

hf_hub_download(
    repo_id=repo_id, 
    filename="config.json",
    local_dir=folder,
    local_dir_use_symlinks=False	
)




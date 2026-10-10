from huggingface_hub import snapshot_download
import torch

# dowonload Qwen/Qwen-Image-2.1-Turbo
snapshot_download(
    repo_id="Qwen/Qwen-Image-2.1-Turbo",
    local_dir="./models/Qwen-Image-2.1-Turbo",
)

# download Qwen/Qwen-Image-2.1-PE-T2I
snapshot_download(
    repo_id="Qwen/Qwen-Image-2.1-PE-T2I",
    local_dir="models/Qwen-Image-2.1-PE-T2I",
)

# download Qwen/Qwen-Image-2.1-PE-I2I
snapshot_download(
    repo_id="Qwen/Qwen-Image-2.1-PE-I2I",
    local_dir="models/Qwen-Image-2.1-PE-I2I",
)

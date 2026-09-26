from huggingface_hub import snapshot_download
import torch

'''
# dowonload Qwen/Qwen-Image-2.1
snapshot_download(
    repo_id=" Qwen/Qwen-Image-2.1",
    local_dir="./models/Qwen-Image-2.1",
)
'''

# download Qwen-Image-2.1-viggle-turbo
# peft and scheduler only
snapshot_download(
    repo_id="Viggle/Qwen-Image-2.1-viggle-turbo",
    allow_patterns=[
        "peft_v0.2.1/*",
        "scheduler/*",
    ],
    local_dir="models/Qwen-Image-2.1-viggle-turbo",
)

'''
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
'''

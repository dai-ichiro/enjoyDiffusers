from huggingface_hub import snapshot_download

'''
# dowonload Qwen/Qwen-Image-2.1
snapshot_download(
    repo_id=" Qwen/Qwen-Image-2.1",
    local_dir="./models/Qwen-Image-2.1",
)
'''

# download OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit 
# text_encoder and transformer only
snapshot_download(
    repo_id="OzzyGT/Qwen_Image_2_1_sdnq_dynamic_4bit",
    allow_patterns=[
        "transformer/*",
        "text_encoder/*",
    ],
    local_dir="models/Qwen_Image_2_1_sdnq_dynamic_4bit",
)

# download OzzyGT/Qwen_Image_2_1_sdnq_dynamic_8bit 
# text_encoder and transformer only
snapshot_download(
    repo_id="OzzyGT/Qwen_Image_2_1_sdnq_dynamic_8bit",
    allow_patterns=[
        "transformer/*",
        "text_encoder/*",
    ],
    local_dir="models/Qwen_Image_2_1_sdnq_dynamic_8bit",
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
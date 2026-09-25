from huggingface_hub import snapshot_download
import torch

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

# transformer quantization
from diffusers import QwenImage21Transformer2DModel
from diffusers import BitsAndBytesConfig as diffusers_config

diffusers_quantization_config = diffusers_config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

transformer = QwenImage21Transformer2DModel.from_pretrained(
    "models/Qwen-Image-2.1",
    subfolder="transformer",
    quantization_config=diffusers_quantization_config,
    dtype=torch.bfloat16
)

transformer.save_pretrained("models/Qwen-Image-2.1_bnb_4bit/transformer")

# text_encoder quantization
from transformers import Qwen3VLForConditionalGeneration
from transformers import BitsAndBytesConfig as transformers_config

transformers_quantization_config = transformers_config(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
    "models/Qwen-Image-2.1",
    subfolder="text_encoder",
    quantization_config=transformers_quantization_config,
    dtype=torch.bfloat16
)

text_encoder.save_pretrained("models/Qwen-Image-2.1_bnb_4bit/text_encoder")
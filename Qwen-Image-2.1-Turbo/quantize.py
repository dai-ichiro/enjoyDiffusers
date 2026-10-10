import torch
from diffusers import SDNQConfig
from transformers import Qwen3VLForConditionalGeneration
from diffusers import QwenImage21Transformer2DModel

## text_encoder (8bit)
text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
    "models/Qwen-Image-2.1-Turbo",
    subfolder="text_encoder",
    quantization_config=SDNQConfig(weights_dtype="int8"),
    dtype=torch.bfloat16
)

SAVE_DIR = "Qwen-Image-2.1-Turbo-sdnq-8bit"

text_encoder.save_pretrained(f"models/{SAVE_DIR}/text_encoder")

## text_encoder (4bit)
text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
    "models/Qwen-Image-2.1-Turbo",
    subfolder="text_encoder",
    quantization_config=SDNQConfig(weights_dtype="int4"),
    dtype=torch.bfloat16
)

SAVE_DIR = "Qwen-Image-2.1-Turbo-sdnq-4bit"

text_encoder.save_pretrained(f"models/{SAVE_DIR}/text_encoder")

## transformer (8bit)
text_encoder=QwenImage21Transformer2DModel.from_pretrained(
    "models/Qwen-Image-2.1-Turbo",
    subfolder="transformer",
    quantization_config=SDNQConfig(weights_dtype="int8"),
    dtype=torch.bfloat16
)

SAVE_DIR = "Qwen-Image-2.1-Turbo-sdnq-8bit"

text_encoder.save_pretrained(f"models/{SAVE_DIR}/transformer")

## transformer (4bit)
text_encoder=QwenImage21Transformer2DModel.from_pretrained(
    "models/Qwen-Image-2.1-Turbo",
    subfolder="transformer",
    quantization_config=SDNQConfig(weights_dtype="int4"),
    dtype=torch.bfloat16
)

SAVE_DIR = "Qwen-Image-2.1-Turbo-sdnq-4bit"

text_encoder.save_pretrained(f"models/{SAVE_DIR}/transformer")

import os
import shutil
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

pup_id = "black-forest-labs/flux-3-image-klein-pup"
local_save_dir = "flux-3-image-klein-pup-bnb-4bit"

os.makedirs(local_save_dir, exist_ok=True)

# 1. 4bit設定の作成
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. プロセッサとモデルの読み込み
processor = AutoProcessor.from_pretrained(pup_id)
upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
    pup_id,
    quantization_config=bnb_config,
    device_map="auto",
).eval()

# 3. ローカルに保存
# モデルとプロセッサを保存
processor.save_pretrained(local_save_dir)
upsampler.save_pretrained(local_save_dir)

# system_prompts.json も同じディレクトリへコピー
system_prompt_path = hf_hub_download(pup_id, "system_prompts.json")
shutil.copy(system_prompt_path, os.path.join(local_save_dir, "system_prompts.json"))

print(f"保存が完了しました: {local_save_dir}")
import gc
import json
import os
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

def upsample_and_unload(
    prompt: str,
    images=(),
    local_save_dir: str = "flux-3-image-klein-pup-bnb-4bit",
) -> str:
    processor = None
    upsampler = None
    try:
        # 1. ローカルからプロセッサとモデルを読み込み
        processor = AutoProcessor.from_pretrained(local_save_dir)
        upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
            local_save_dir,
            device_map="auto",
        ).eval()

        # 2. プロンプトテンプレートの準備
        with open(os.path.join(local_save_dir, "system_prompts.json"), "r", encoding="utf-8") as f:
            system_prompts = json.load(f)

        if images:
            text = (
                "Reference images follow in order (the first image, the second image, ...).\n"
                f"User request: {prompt}\nOutput the caption."
            )
            content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}]
            messages = [{"role": "system", "content": system_prompts["i2i"]}, {"role": "user", "content": content}]
        else:
            messages = [{"role": "system", "content": system_prompts["t2i"]}, {"role": "user", "content": prompt}]

        # 3. 入力トークナイズ & 推論
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(upsampler.device)

        with torch.no_grad():
            output = upsampler.generate(**inputs, max_new_tokens=3584, do_sample=False)

        result_text = processor.decode(
            output[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # 生成結果以外の中間テンソル参照を破棄
        del inputs, output

        return result_text

    finally:
        # 4. エラー発生時でも確実にVRAMとシステムメモリを解放
        del upsampler
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
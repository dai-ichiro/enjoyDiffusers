import gradio as gr
import numpy as np
import random
import gc
import uuid
from datetime import datetime

from PIL import Image

import os
import json

import sdnq  # noqa: F401  (registers the SDNQ quantized-weight backend with transformers/diffusers)
import torch
from diffusers import ModularPipeline, Flux3Transformer2DModel
from diffusers.hooks import apply_group_offloading
from transformers import Qwen3VLForConditionalGeneration

from upsampler_module import upsample_and_unload

# ============== Configuration ==============
# Log directory, customizable via the LOG_DIR environment variable
LOG_DIR = os.environ.get("LOG_DIR", "./generation_logs_paper_case")

# ============== Local model configuration ==============
BASE_MODEL_ID = "black-forest-labs/flux-3-image-klein"

# Maps the UI label to the quantized model repo holding its
# text_encoder/transformer subfolders. Unquantized models are too large
# and are intentionally not offered.
QUANT_CHOICES = {
    "sdnq-4bit": "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_4bit",
    "sdnq-8bit": "OzzyGT/Flux_3_Image_Klein_sdnq_dynamic_8bit",
}
DEFAULT_QUANT_LABEL = os.environ.get("QUANT_CHOICE", "sdnq-4bit")
if DEFAULT_QUANT_LABEL not in QUANT_CHOICES:
    DEFAULT_QUANT_LABEL = "sdnq-4bit"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

MAX_INPUT_IMAGES = 10

# Output size limits (pixels). Sizes are rounded to a multiple of SIZE_STEP.
DEFAULT_SIZE = 1024
MIN_SIZE = 256
MAX_SIZE = 2048
SIZE_STEP = 16


# ============== Logging class ==============
class GenerationLogger:
    """Logs each generation: JSON (prompt info) + PNG (image)"""

    def __init__(self, log_dir="./logs"):
        """
        Initialize the logger

        Args:
            log_dir: directory to store logs in
        """
        self.log_dir = log_dir
        self._ensure_dir()


    def _ensure_dir(self):
        """Ensure the log directory exists"""
        os.makedirs(self.log_dir, exist_ok=True)

    def set_log_dir(self, log_dir):
        """
        Change the log directory at runtime

        Args:
            log_dir: the new log directory path
        """
        self.log_dir = log_dir
        self._ensure_dir()

    def log_generation(self, original_prompt, enhanced_prompt, image, seed, params, gpu_id=None, input_images=None, api_response=None):
        """
        Save a complete record of one generation

        Args:
            original_prompt: the user's original prompt
            enhanced_prompt: the prompt after LLM rewriting
            image: the generated PIL image object
            seed: the seed used
            params: dict of generation parameters
            gpu_id: the GPU ID used
            input_images: list of input images (PIL Image objects)

        Returns:
            str: base name of the log files (without extension)
        """
        self._ensure_dir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"{timestamp}_{unique_id}"

        # Save input images
        input_image_paths = []
        if input_images and len(input_images) > 0:
            for i, img in enumerate(input_images):
                input_path = os.path.join(self.log_dir, f"{base_name}.{i}.png")
                img.save(input_path)
                input_image_paths.append(os.path.abspath(input_path))
            print(f"[Logger] Saved {len(input_images)} input image(s)")

        # Save the output image
        image_path = os.path.join(self.log_dir, f"{base_name}.png")
        image.save(image_path)

        # Save the JSON log
        json_path = os.path.join(self.log_dir, f"{base_name}.json")
        log_data = {
            "timestamp": timestamp,
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "seed": seed,
            "gpu_id": gpu_id,
            "parameters": params,
            "input_images": input_image_paths,
            "image_path": os.path.abspath(image_path),
            "api_response": api_response
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"[Logger] Log saved: {base_name}.json / {base_name}.png")
        return base_name



def validate_image_count(images):
    if images is not None and len(images) > MAX_INPUT_IMAGES:
        raise gr.Error("Up to 10 input images are supported.")


# ============== Local model loading ==============
# The diffusion pipeline is kept resident for the life of the process. The
# prompt upsampler is loaded/unloaded per request (see upsampler_module.py),
# so its VRAM is freed before the diffusion pipeline runs.
_pipe = None
_pipe_quant_label = None


def free_vram():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def _build_pipeline(quant_label):
    repo = QUANT_CHOICES[quant_label]
    blocks = ModularPipeline.from_pretrained(BASE_MODEL_ID).blocks
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(BASE_MODEL_ID)

    pipe.update_components(
        transformer=Flux3Transformer2DModel.from_pretrained(
            repo, subfolder="transformer", dtype=DTYPE
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            repo, subfolder="text_encoder", dtype=DTYPE
        ),
    )
    pipe.load_components(dtype=DTYPE)

    if DEVICE == "cuda":
        offload = {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "use_stream": True,
            "low_cpu_mem_usage": True,
        }
        apply_group_offloading(pipe.transformer, offload_type="leaf_level", **offload)
        apply_group_offloading(pipe.text_encoder, offload_type="leaf_level", **offload)
        pipe.vae.to("cuda")
    return pipe


def get_pipeline(quant_label):
    """Return the resident diffusion pipeline, rebuilding it when the
    quantization variant changes."""
    global _pipe, _pipe_quant_label

    if quant_label not in QUANT_CHOICES:
        quant_label = DEFAULT_QUANT_LABEL

    if _pipe is not None and quant_label != _pipe_quant_label:
        print(f"[Local] Switching quantization: {_pipe_quant_label} -> {quant_label} ...")
        _pipe = None
        free_vram()

    if _pipe is None:
        print(f"[Local] Loading diffusion pipeline ({QUANT_CHOICES[quant_label]}) ...")
        _pipe = _build_pipeline(quant_label)
        _pipe_quant_label = quant_label
        print("[Local] Diffusion pipeline ready.")

    return _pipe


def rewrite_prompt(instruction_text, ref_images=None):
    """Run the local prompt upsampler (t2i without images, i2i with images)."""
    result = upsample_and_unload(instruction_text, ref_images or ())
    free_vram()
    return result


def run_local_generation(prompt, mode, images, height, width, seed, quant_label):
    """Run the local diffusion pipeline and return a PIL.Image."""
    pipe = get_pipeline(quant_label)
    generator = torch.Generator().manual_seed(int(seed))

    kwargs = {}
    if mode == "edit":
        kwargs["image"] = images
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            output="images",
            **kwargs,
        )[0]
    free_vram()

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


# Initialize the logger
print(f"Initializing logger with directory: {LOG_DIR}")
logger = GenerationLogger(log_dir=LOG_DIR)


# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Stage 2: Image Generation Function ---
def generate_image_stage(
    input_images,
    original_prompt,
    log_dir,
    seed=42,
    randomize_seed=False,
    prompt_extend=True,
    quant_label=DEFAULT_QUANT_LABEL,
    width=DEFAULT_SIZE,
    height=DEFAULT_SIZE,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image locally.
    Mode is auto-detected from whether input images are present: images -> Edit, none -> T2I.
    When prompt_extend is enabled, the local prompt upsampler rewrites the prompt first.

    Returns (image, seed, rewritten_prompt).
    """
    validate_image_count(input_images)

    # Update the log directory if the user changed it
    if log_dir and log_dir.strip():
        logger.set_log_dir(log_dir.strip())

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Load input images into PIL Images from gallery
    pil_images = []
    if input_images is not None and len(input_images) > 0:
        for item in input_images:
            try:
                if isinstance(item, Image.Image):
                    pil_images.append(item.convert("RGB"))
                elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item, str):
                    pil_images.append(Image.open(item).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception as e:
                print(f"[Warning] Failed to load input image: {e}")
                continue
        print(f"[Local] Loaded {len(pil_images)} input images for editing")

    # Auto-detect the mode based on whether input images are present
    is_edit_mode = len(pil_images) > 0
    mode = "edit" if is_edit_mode else "t2i"

    print(f"[Local] Mode: {mode} (auto-detected: {'has input images' if is_edit_mode else 'no input images'})")
    print(f"[Local] Prompt: '{original_prompt[:100]}...'" if len(original_prompt) > 100 else f"[Local] Prompt: '{original_prompt}'")
    print(f"[Local] Prompt Extend: {prompt_extend}")
    print(f"[Local] Input images count: {len(pil_images)}")

    # Run the local prompt-rewriting model, if enabled
    rewritten_prompt = ""
    gen_prompt = original_prompt
    if prompt_extend:
        print("[Local] Rewriting prompt ...")
        rewritten_prompt = rewrite_prompt(original_prompt, ref_images=pil_images if is_edit_mode else None).strip()
        gen_prompt = rewritten_prompt or original_prompt

    # Clamp to the allowed range and round to a multiple the model can handle.
    gen_width = min(MAX_SIZE, max(MIN_SIZE, round(int(width or DEFAULT_SIZE) / SIZE_STEP) * SIZE_STEP))
    gen_height = min(MAX_SIZE, max(MIN_SIZE, round(int(height or DEFAULT_SIZE) / SIZE_STEP) * SIZE_STEP))

    print(f"[Local] Seed: {seed}, Size: {gen_width}x{gen_height}")

    # Run local inference
    image = run_local_generation(
        prompt=gen_prompt,
        mode=mode,
        images=pil_images if len(pil_images) > 0 else None,
        height=gen_height,
        width=gen_width,
        seed=seed,
        quant_label=quant_label,
    )

    # Log this generation
    params = {
        "mode": mode,
        "height": gen_height,
        "width": gen_width,
        "prompt_extend": prompt_extend,
        "quantization": quant_label,
        "input_images_count": len(pil_images),
    }

    log_name = logger.log_generation(
        original_prompt=original_prompt,
        enhanced_prompt=gen_prompt,
        image=image,
        seed=seed,
        params=params,
        gpu_id=None,
        input_images=pil_images if len(pil_images) > 0 else None,
    )

    print(f"[Local] Generation complete, logged as: {log_name}")

    return image, seed, rewritten_prompt


def make_placeholder_image(text, width=512, height=320, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
    """Generate a placeholder image with centered text"""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGBA", (width, height), (*bg_color, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    return img


# --- Unified generate_with_enhance generator ---
def generate_with_enhance(
    input_images,
    original_prompt,
    enable_extend,
    log_dir,
    seed,
    randomize_seed,
    quant_label,
    width,
    height,
):
    """
    Generate an image locally; prompt_extend runs the local rewriting model.
    Yields intermediate results so the UI updates progressively.
    """
    yield make_placeholder_image("Generating image..."), seed, ""

    image, seed, rewritten_prompt = generate_image_stage(
        input_images, original_prompt,
        log_dir, seed, randomize_seed,
        prompt_extend=enable_extend,
        quant_label=quant_label,
        width=width,
        height=height,
    )
    yield image, seed, rewritten_prompt


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(title="FLUX.3 Klein") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("<h1 style='text-align: center;'>FLUX.3 Klein</h1>")
        instructions = gr.Markdown("""
## FLUX.3 Klein Demo
1. Generate from text without an input image, or upload 1–10 images to edit or combine them. Refer to images by their upload order in your prompt.
2. Prompt enhancement is enabled by default. Turn it off to use your original prompt directly.
""")
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(
                    label="Input Images (for editing)",
                    show_label=True,
                    type="pil",
                    interactive=True,
                    columns=5,
                    height="auto"
                )
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, type="pil", image_mode="RGBA", format='png')

        # Prompt Input
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    lines=6,
                    max_lines=24,
                    show_label=True,
                    placeholder="Describe what you want to generate or edit...",
                    container=True,
            )
        with gr.Row():
            enable_extend = gr.Checkbox(
                label="Enable Prompt Extend",
                value=True,
            )
            generate_button = gr.Button("Generate Image", variant="primary")

        rewritten_prompt_output = gr.Textbox(
            value="", lines=4, max_lines=20, interactive=False,
            label="Rewritten prompt",
            placeholder="After generation with prompt enhancement enabled, the local rewrite appears here. Blank if none is returned.",
        )
        enable_extend.change(fn=lambda: "", inputs=[], outputs=[rewritten_prompt_output], queue=False)



        with gr.Accordion("Advanced Settings", open=False) as advanced:
            # Log directory configuration
            log_dir_input = gr.Textbox(
                label="Log Directory",
                show_label=True,
                value=LOG_DIR,
                placeholder="Enter a log directory path, e.g. ./generation_logs",
                interactive=True,
                visible=False,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width_input = gr.Number(
                    label="Width",
                    value=DEFAULT_SIZE,
                    minimum=MIN_SIZE,
                    maximum=MAX_SIZE,
                    step=SIZE_STEP,
                    precision=0,
                )
                height_input = gr.Number(
                    label="Height",
                    value=DEFAULT_SIZE,
                    minimum=MIN_SIZE,
                    maximum=MAX_SIZE,
                    step=SIZE_STEP,
                    precision=0,
                )

            quant_choice_input = gr.Radio(
                label="Quantization",
                choices=list(QUANT_CHOICES.keys()),
                value=DEFAULT_QUANT_LABEL,
                info="Switching reloads the transformer/text_encoder.",
            )

    input_images.label = "Input images (editing, up to 10)"
    result.label = "Result"
    prompt.label = "Prompt"
    prompt.placeholder = "Describe what to generate or edit; refer to images 1–10 in upload order…"
    enable_extend.label = "Enhance prompt"
    generate_button.value = "Generate image"
    advanced.label = "Advanced settings"
    input_images.upload(fn=validate_image_count, inputs=[input_images], outputs=[], queue=False)

    # Generate Image button event (with optional local prompt enhancement)
    generate_button.click(
        fn=generate_with_enhance,
        inputs=[
            input_images,     # input images (present -> Edit mode, absent -> T2I mode)
            prompt,           # original_prompt
            enable_extend,    # whether to run the local prompt-rewriting model
            log_dir_input,    # log_dir
            seed,
            randomize_seed,
            quant_choice_input,     # quantization variant
            width_input,
            height_input,
        ],
        outputs=[result, seed, rewritten_prompt_output],
        concurrency_limit=1,  # a single local GPU can only run one generation at a time
    )

# A single local GPU pipeline can't serve concurrent generations; keep the queue serialized.
demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":

    print("\n[Notice] ブラウザからは http://localhost:7860 で開いてください。\n")
    
    demo.launch(
        css=css,
        server_name="0.0.0.0",
        share=False,
        quiet=True
    )


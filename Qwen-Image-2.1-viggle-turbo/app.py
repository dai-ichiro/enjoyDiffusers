import gradio as gr
import numpy as np
import random
import gc
import uuid
from datetime import datetime

from PIL import Image

import os
import json

import torch
from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

# ============== Configuration ==============
# Log directory, customizable via the LOG_DIR environment variable
LOG_DIR = os.environ.get("LOG_DIR", "./generation_logs_paper_case")

# ============== Local model configuration ==============
MODELS_DIR = os.environ.get(
    "MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
)

BASE_MODEL_DIR = os.path.join(MODELS_DIR, "Qwen-Image-2.1")
LORA_MODEL_DIR = os.path.join(MODELS_DIR, "Qwen-Image-2.1-viggle-turbo")

# Prompt-rewriting ("prompt extend") models
PE_T2I_DIR = os.path.join(MODELS_DIR, "Qwen-Image-2.1-PE-T2I")
PE_I2I_DIR = os.path.join(MODELS_DIR, "Qwen-Image-2.1-PE-I2I")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Viggle Turbo schedule configuration
STEPS = 6
SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]

MAX_INPUT_IMAGES = 3


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
        raise gr.Error("Up to 3 input images are supported.")


# ============== Local model loading ==============
_pipe = None


def free_vram():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def get_pipeline():
    """Return the resident diffusion pipeline with Viggle Turbo LoRA and scheduler loaded."""
    global _pipe

    if _pipe is None:
        print(f"[Local] Loading diffusion pipeline (base: {BASE_MODEL_DIR}, lora: {LORA_MODEL_DIR}) ...")
        _pipe = QwenImage21Pipeline.from_pretrained(
            BASE_MODEL_DIR, dtype=DTYPE
        )
        print("[Local] Loading Viggle Turbo LoRA adapter ...")
        _pipe.transformer.load_lora_adapter(
            LORA_MODEL_DIR,
            subfolder="peft_v0.2.1",
            weight_name="adapter_model.safetensors",
            prefix=None,
        )
        print("[Local] Setting FlowMatchEulerDiscreteScheduler ...")
        _pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            LORA_MODEL_DIR,
            subfolder="scheduler",
        )
        if DEVICE == "cuda":
            _pipe.enable_model_cpu_offload()
        else:
            _pipe.to(DEVICE)
        print("[Local] Diffusion pipeline with Viggle Turbo ready.")

    return _pipe


def ensure_pipeline_offloaded():
    """Make sure the resident diffusion pipeline isn't holding VRAM before a
    separate VRAM-hungry model (the prompt enhancer) is loaded."""
    if _pipe is not None and DEVICE == "cuda":
        _pipe.maybe_free_model_hooks()
    free_vram()


def rewrite_prompt(instruction_text, ref_images=None):
    """
    Run the local prompt-rewriting ("prompt extend") model.

    - ref_images given (list of PIL Images): uses Qwen-Image-2.1-PE-I2I (editing).
    - ref_images is None: uses Qwen-Image-2.1-PE-T2I (text-to-image).

    Returns: dict with "rewritten_prompt".
    """
    ensure_pipeline_offloaded()

    if ref_images:
        model_dir = PE_I2I_DIR
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=DTYPE, device_map="auto"
        ).eval()

        with open(os.path.join(model_dir, "system_prompt.txt"), encoding="utf-8") as f:
            system_prompt = f.read().strip()

        image_contents = [{"type": "image", "image": img} for img in ref_images]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": image_contents + [{"type": "text", "text": instruction_text}]},
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=True,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=24000,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
            )
        gen = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        del processor, model, inputs, out, image_contents
        free_vram()
    else:
        model_dir = PE_T2I_DIR
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=DTYPE, device_map="auto"
        ).eval()

        with open(os.path.join(model_dir, "system_prompt.txt"), encoding="utf-8") as f:
            system_prompt = f.read().strip()

        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=16256,
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
            )
        gen = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        del tokenizer, model, inputs, out
        free_vram()

    _, _, answer = gen.partition("</think>")
    return json.loads(answer.strip())


def run_local_generation(prompt, mode, images, height, width, seed, use_vae_tiling):
    """Run the local diffusion pipeline with Viggle Turbo parameters and return a PIL.Image."""
    pipe = get_pipeline()
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    if use_vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    else:
        pipe.vae.disable_tiling()
        pipe.vae.disable_slicing()

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=images if mode == "edit" else None,
            height=height,
            width=width,
            num_inference_steps=STEPS,
            sigmas=SIGMAS,
            generator=generator,
        )
    free_vram()

    image = result.images[0]
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
    size_choice="small",
    use_vae_tiling=True,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate an image locally using Viggle Turbo LoRA.
    Mode is auto-detected from whether input images are present: images -> Edit, none -> T2I.
    When prompt_extend is enabled, prompt rewriting runs first, but the final image resolution is
    always set by user selection (`size_choice`).
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

    # Auto-detect mode
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
        rewrite_result = rewrite_prompt(original_prompt, ref_images=pil_images if is_edit_mode else None)
        rewritten_prompt = (rewrite_result.get("rewritten_prompt") or "").strip()
        gen_prompt = rewritten_prompt or original_prompt

    # Size selection based on mode and user size_choice choice
    if mode == "t2i":
        gen_width, gen_height = (2048, 2048) if size_choice == "large" else (1024, 1024)
    else:
        gen_width, gen_height = (1536, 1536) if size_choice == "large" else (1024, 1024)

    print(f"[Local] Seed: {seed}, Size: {gen_width}x{gen_height} ({size_choice}), Steps: {STEPS}")

    # Run local inference with Viggle Turbo
    image = run_local_generation(
        prompt=gen_prompt,
        mode=mode,
        images=pil_images if len(pil_images) > 0 else None,
        height=gen_height,
        width=gen_width,
        seed=seed,
        use_vae_tiling=use_vae_tiling,
    )

    # Log generation
    params = {
        "mode": mode,
        "height": gen_height,
        "width": gen_width,
        "size_choice": size_choice,
        "vae_tiling": use_vae_tiling,
        "prompt_extend": prompt_extend,
        "input_images_count": len(pil_images),
        "num_inference_steps": STEPS,
        "sigmas": SIGMAS,
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
    size_choice,
    use_vae_tiling,
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
        size_choice=size_choice,
        use_vae_tiling=use_vae_tiling,
    )
    yield image, seed, rewritten_prompt


def make_example_loader(images, text, extend):
    """Bind each example to a zero-argument callback without late binding."""
    def load_example():
        return list(images), text, extend
    return load_example


ENGLISH_EXAMPLES = [
    ("Editorial portrait", "Create an editorial portrait of a botanist in a sunlit greenhouse, surrounded by ferns and delicate orchids. Natural skin texture, linen clothing, soft morning backlight, subtle film grain, medium-format photography, calm expression, no text or watermark."),
    ("Typography poster", 'Design a refined travel poster for a fictional night train. Render the headline exactly as "THE MIDNIGHT EXPRESS" and the subtitle "A journey under the stars". A silver train curves through dark blue mountains beneath a crescent moon. Art Deco geometry, ivory and gold lettering, clear typographic hierarchy, generous margins, print-ready composition.'),
    ("Six-panel storyboard", "Create a six-panel cinematic storyboard about a small robot restoring an abandoned rooftop garden. Show: arrival at dawn, discovery of a dried seedling, repairing an irrigation pipe, planting new seeds, the first rain, and a lush garden at sunset. Keep the robot's round yellow body and blue eyes consistent in every panel. Clear panel borders, expressive visual storytelling, detailed environments, no captions."),
    ("Product photography", "Photograph a translucent emerald perfume bottle on pale limestone beside a shallow pool. Rippling sunlight reflects through the glass onto the stone. A single olive branch frames the upper left corner. Luxury product photography, realistic refraction, crisp bottle edges, soft shadows, uncluttered composition, no logo or text."),
]


# --- Examples and UI Layout ---
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
DEMO_CASES = []
GENERATED_REFERENCES = {}
cases_path = os.path.join(EXAMPLE_DIR, "cases.json")
references_path = os.path.join(EXAMPLE_DIR, "generated_references.json")
if os.path.isfile(cases_path):
    with open(cases_path, encoding="utf-8") as case_file:
        DEMO_CASES = json.load(case_file)
if os.path.isfile(references_path):
    with open(references_path, encoding="utf-8") as reference_file:
        GENERATED_REFERENCES = {case["prompt"]: case for case in json.load(reference_file)}


def add_generated_reference(prompt):
    reference = GENERATED_REFERENCES.get(prompt)
    if not reference:
        return
    gr.Gallery(
        value=[os.path.join(EXAMPLE_DIR, name) for name in reference["outputs"]],
        columns=1, height=280, interactive=False,
        label="Reference output",
    )


def add_case_examples(cases, input_images, prompt, enable_extend):
    """Show bundled reference images and load full prompts in their original order."""
    for case in cases:
        paths = [os.path.join(EXAMPLE_DIR, name) for name in case["inputs"]]
        references = [os.path.join(EXAMPLE_DIR, name) for name in case["outputs"]]
        with gr.Accordion(case["title_en"], open=False) as panel:
            with gr.Row():
                if paths:
                    gr.Gallery(
                        value=[(path, str(index)) for index, path in enumerate(paths, 1)],
                        columns=min(len(paths), 5), height=280, interactive=False,
                        label="Input references (in numbered order)",
                    )
                if references:
                    gr.Gallery(value=references, columns=1, height=280, interactive=False,
                               label="Reference output")
            gr.Textbox(value=case["prompt"], lines=4, max_lines=20, interactive=False,
                       label="Full prompt")
            button = gr.Button("Use this example")
            button.click(
                fn=make_example_loader(paths, case["prompt"], case["prompt_extend"]),
                inputs=[], outputs=[input_images, prompt, enable_extend], queue=False,
            )


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(title="Qwen Image 2.1 Viggle Turbo Demo") as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("<h1 style='text-align: center;'>Qwen-Image 2.1 Viggle Turbo</h1>")
        instructions = gr.Markdown("""
## Qwen Image 2.1 Viggle Turbo Demo
1. Generate from text without an input image, or upload 1–3 images to edit or combine them. Refer to images by their upload order in your prompt.
2. Viggle Turbo LoRA is applied automatically for high-speed generation (6 steps).
3. Prompt enhancement is enabled by default. Turn it off to use your original prompt directly.
4. Image size is decided by user selection below ("small" or "large").
5. For transparent image generation, use the following prompt format and replace `xxxxx` with your image description:

   `This is an RGBA image with transparency. xxxxx The image has alpha channel and the background is transparent.`
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

            size_choice_input = gr.Radio(
                label="Image Size",
                choices=["small", "large"],
                value="small",
                info="Text-to-image: small (1024x1024), large (2048x2048) | Editing: small (1024x1024), large (1536x1536)",
            )

            vae_tiling_input = gr.Checkbox(
                label="VAE tiling/slicing",
                value=True,
                info="Decodes the image piece by piece to keep peak VRAM down. Needed for "
                     "large sizes; can be turned off for small ones without running out of VRAM.",
            )

        # --- Examples ---
        if DEMO_CASES or ENGLISH_EXAMPLES:
            gr.Markdown("### Examples")

        if DEMO_CASES:
            gr.Markdown("**Featured text-to-image examples**")
            add_case_examples([case for case in DEMO_CASES if not case["inputs"]],
                              input_images, prompt, enable_extend)
            gr.Markdown("**Image-editing examples**")
            add_case_examples([case for case in DEMO_CASES if case["inputs"]],
                              input_images, prompt, enable_extend)

        if ENGLISH_EXAMPLES:
            gr.Markdown("**Additional text-to-image examples**")
            for title, text in ENGLISH_EXAMPLES:
                with gr.Accordion(title, open=False):
                    add_generated_reference(text)
                    gr.Markdown(text)
                    button = gr.Button("Use this prompt")
                    button.click(fn=make_example_loader([], text, True), inputs=[],
                                 outputs=[input_images, prompt, enable_extend], queue=False)

    input_images.label = "Input images (editing, up to 3)"
    result.label = "Result"
    prompt.label = "Prompt"
    prompt.placeholder = "Describe what to generate or edit; refer to images 1–3 in upload order…"
    enable_extend.label = "Enhance prompt"
    generate_button.value = "Generate image"
    advanced.label = "Advanced settings"
    input_images.upload(fn=validate_image_count, inputs=[input_images], outputs=[], queue=False)

    # Generate Image button event
    generate_button.click(
        fn=generate_with_enhance,
        inputs=[
            input_images,        # input images (present -> Edit mode, absent -> T2I mode)
            prompt,              # original_prompt
            enable_extend,       # whether to run local prompt-rewriting model
            log_dir_input,       # log_dir
            seed,
            randomize_seed,
            size_choice_input,   # image size ("small" / "large")
            vae_tiling_input,    # whether to decode VAE output in tiles/slices
        ],
        outputs=[result, seed, rewritten_prompt_output],
        concurrency_limit=1,
    )

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    _script_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    _allowed = [
        os.path.join(_script_dir, "examples"),
        os.path.realpath(EXAMPLE_DIR),
    ]
    demo.launch(
        allowed_paths=_allowed,
        css=css,
        share=False
    )

from accelerate import Accelerator
from diffusers import DiffusionPipeline
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    required=True,
    type=int,
    help='checkpoint',
)
parser.add_argument(
    '--model',
    required=True,
    type=str,
    help='model',
)
opt = parser.parse_args()

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(model_id)

accelerator = Accelerator()

# Use text_encoder if `--train_text_encoder` was used for the initial training
unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# Restore state from a checkpoint path. You have to use the absolute path here.
checkpoint_path = os.path.join(opt.model, f'checkpoint-{opt.checkpoint}')
accelerator.load_state(checkpoint_path)

# Rebuild the pipeline with the unwrapped models (assignment to .unet and .text_encoder should work too)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
)

# Perform inference, or save, or push to the hub
pipeline.save_pretrained(f"{opt.model}_{opt.checkpoint}")
from diffusers import StableDiffusionPipeline
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    required=True,
    type=str,
    help='model id'
)
parser.add_argument(
    '--seed',
    type=int,
    default=200,
    help='seed'
)
parser.add_argument(
    '--width',
    type=int,
    default=512,
    help='width'
)
parser.add_argument(
    '--height',
    type=int,
    default=512,
    help='height'
)
opt = parser.parse_args()

model_id = opt.model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    safety_checker=None).to("cuda")

prompt = "A photo of sks robo on the beach"

seed = opt.seed

for i in range(4):
    new_seed = seed + i
    generator = torch.Generator(device="cuda").manual_seed(new_seed)
    image = pipe(
        prompt = prompt, 
        num_inference_steps = 50,
        generator = generator,
        num_images_per_prompt = 1,
        width = opt.width,
        height = opt.height
        ).images[0]
    image.save(f'{model_id}_{new_seed}.png')

from diffusers import VersatileDiffusionImageVariationPipeline
import torch
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--scale',
    nargs='*',
    default=[7.5],    
    type=float,
    help='guidance_scale',
)
opt = parser.parse_args()
original_image = opt.image
init_image = Image.open(original_image).convert("RGB").resize((512, 512))

pipe = VersatileDiffusionImageVariationPipeline.from_pretrained(
    "versatile-diffusion", torch_dtype=torch.float32
)
pipe.to("cuda")

os.makedirs('results', exist_ok=True)

scale_list = opt.scale

for i in range(opt.n_samples):
    seed = opt.seed + i
    for scale in scale_list:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            image = init_image,
            guidance_scale = scale,
            generator=generator).images[0]
        image.save(os.path.join('results', f'variation_seed{seed}_scale{scale}.png'))

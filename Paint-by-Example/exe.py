from PIL import Image
import torch
from diffusers import DiffusionPipeline

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--image', type=str, help='original image' )
parser.add_argument('--mask', type=str, help='mask image' )
parser.add_argument('--example', type=str, help='example image' )
parser.add_argument('--resolution', type=int, default=512, help='resolution')

opt = parser.parse_args()

img_fname = opt.image
mask_fname = opt.mask
example_fname = opt.example
resolution = opt.resolution

init_image = Image.open(img_fname).convert('RGB').resize((resolution, resolution))
mask_image = Image.open(mask_fname).convert('RGB').resize((resolution, resolution))
example_image = Image.open(example_fname).convert('RGB').resize((resolution, resolution))

pipe = DiffusionPipeline.from_pretrained(
    'Paint-by-Example',
    torch_dtype=torch.float32).to("cuda")

n_samples = 5

for i in range(n_samples):
    image = pipe(
        image=init_image,
        mask_image=mask_image,
        width = resolution,
        height = resolution,
        example_image=example_image).images[0]

    image.save(f'result_{i}.png')
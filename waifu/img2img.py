import os
import argparse
import datetime
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=200,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=5,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--negative_prompt',
    action="store_true",
    help='if enabled, use negative prompt',
)
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
opt = parser.parse_args()

original_image = opt.image
init_image = Image.open(original_image).convert("RGB").resize((512, 512))

if os.path.isfile('prompt.txt'):
    print('reading prompts from prompt.txt')
    with open('prompt.txt', 'r') as f:
        prompt = f.read().splitlines()
        prompt = ','.join(prompt)
else:
    prompt = 'anime of tsundere moe kawaii beautiful girl'

if opt.negative_prompt and os.path.isfile('negative_prompt.txt'):
    print('reading negative prompts from negative_prompt.txt')
    with open('negative_prompt.txt', 'r') as f:
        negative_prompt = f.read().splitlines()
        negative_prompt = ','.join(negative_prompt)
else:
    negative_prompt = None

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

model_id = "./waifu-diffusion-v1-3-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

os.makedirs('results', exist_ok=True)

now = datetime.datetime.today()
now_str = now.strftime('%m%d_%H%M')

for i in range(opt.n_samples):
    seed  = opt.seed + i
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt = prompt,
        negative_prompt = negative_prompt,
        image = init_image,
        generator = generator,
        num_images_per_prompt = 1).images[0]
    image.save(os.path.join('results', f'{now_str}_seed{seed}.png'))

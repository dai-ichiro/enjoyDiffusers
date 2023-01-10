import os
import argparse
import datetime
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

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
    default=1,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--scale',
    nargs="*",
    default=[7.5],    
    type=float,
    help='guidance_scale',
)
parser.add_argument(
    '--steps',
    type=int,
    default=50,
    help='num_inference_steps',
)
parser.add_argument(
    '--negative_prompt',
    action="store_true",
    help='if enabled, use negative prompt',
)
parser.add_argument(
    '--scheduler',
    type=str,
    default='pndm',
    choices=['pndm', 'multistepdpm', 'eulera']
)
opt = parser.parse_args()

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

model_id = "./waifu-diffusion"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
scheduler = opt.scheduler
match scheduler:
    case 'multistepdpm':
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    case 'eulera':
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    case _:
        None
pipe = pipe.to("cuda")

def null_safety(images, **kwargs):
    return images, False
 
pipe.safety_checker = null_safety

os.makedirs('results', exist_ok=True)

now = datetime.datetime.today()
now_str = now.strftime('%m%d_%H%M')

scale_list = opt.scale

steps = opt.steps

for i in range(opt.n_samples):
    for scale in scale_list:
        seed  = opt.seed + i
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            generator = generator,
            guidance_scale = scale,
            num_inference_steps = steps,
            num_images_per_prompt = 1).images[0]
        image.save(os.path.join('results', f'{now_str}_{scheduler}_seed{seed}_scale{scale}_steps{steps}.png'))
import os
import sys
import glob
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
parser.add_argument(
    '--strength',
    nargs='*',
    default=[0.8],
    type=float,
    help='strength',
)
parser.add_argument(
    '--steps',
    nargs='*',
    default=[50],
    type=int,
    help='num_inference_steps',
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
parser.add_argument(
    '--scheduler',
    type=str,
    default='pndm',
    choices=['pndm', 'multistepdpm', 'eulera']
)
opt = parser.parse_args()

globresult = glob.glob('*')
dirlist =[]
for file_or_dir in globresult:
    if os.path.isdir(file_or_dir) and file_or_dir != 'results':
        dirlist.append(file_or_dir)

if len(dirlist) == 1:
    model_id = dirlist[0]
    print(f'model id: {model_id}')
else:
    print('Unable to identify model')
    sys.exit()

original_image = opt.image
init_image = Image.open(original_image).convert("RGB").resize((512, 512))

if os.path.isfile('prompt.txt'):
    print('reading prompts from prompt.txt')
    with open('prompt.txt', 'r') as f:
        #prompt = f.read().splitlines()
        prompt = f.readlines()
        prompt = [x.strip() for x in prompt]
        prompt = ','.join(prompt)
else:
    print('Unable to find prompt.txt')
    sys.exit()

if opt.negative_prompt and os.path.isfile('negative_prompt.txt'):
    print('reading negative prompts from negative_prompt.txt')
    with open('negative_prompt.txt', 'r') as f:
        #negative_prompt = f.read().splitlines()
        negative_prompt = f.readlines()
        negative_prompt = [x.strip() for x in negative_prompt]
        negative_prompt = ','.join(negative_prompt)
else:
    negative_prompt = None

print(f'prompt: {prompt}')
print(f'negative prompt: {negative_prompt}')

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
scheduler = opt.scheduler
match scheduler:
    case 'pmdn':
        from diffusers import  PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
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
strength_list = opt.strength
steps_list = opt.steps

for i in range(opt.n_samples):
    seed  = opt.seed + i
    for scale in scale_list:
        for strength in strength_list:
            for steps in steps_list:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(
                    prompt = prompt,
                    negative_prompt = negative_prompt,
                    image = init_image,
                    generator = generator,
                    guidance_scale = scale,
                    strength = strength,
                    num_inference_steps = steps,
                    num_images_per_prompt = 1).images[0]
                image.save(os.path.join('results', f'{now_str}_{scheduler}_seed{seed}_scale{scale}_strength{strength}_steps{steps}.png'))
import os
import argparse
import datetime
import torch
from PIL import Image
from copy import deepcopy
from diffusers import StableDiffusionDepth2ImgPipeline

parser = argparse.ArgumentParser()

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
    '--newword',
    type=str,
    nargs="*",
    help='add new words'
)
opt = parser.parse_args()

original_image = opt.image

init_image = Image.open(original_image).convert("RGB").resize((512, 512))

if os.path.isfile('prompt.txt'):
    print('reading prompts from prompt.txt')
    with open('prompt.txt', 'r') as f:
        #prompt = f.read().splitlines()
        prompt = f.readlines()
        prompt = [x.strip() for x in prompt if x.strip() != '']
else:
    prompt = ['anime of tsundere moe kawaii beautiful girl']

if opt.negative_prompt and os.path.isfile('negative_prompt.txt'):
    print('reading negative prompts from negative_prompt.txt')
    with open('negative_prompt.txt', 'r') as f:
        #negative_prompt = f.read().splitlines()
        negative_prompt = f.readlines()
        negative_prompt = [x.strip() for x in negative_prompt if x.strip() != '']
        negative_prompt = ','.join(negative_prompt)
else:
    negative_prompt = None

print(f'negative prompt: {negative_prompt}')

model_id = "./stable-diffusion-2-depth"

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

def null_safety(images, **kwargs):
    return images, False
 
pipe.safety_checker = null_safety

os.makedirs('results', exist_ok=True)

now = datetime.datetime.today()
now_str = now.strftime('%m%d_%H%M')

prompt_list = []
if opt.newword:
    newword_list = opt.newword
    newword = ' '.join(newword_list)
    
    for i in range(len(prompt) + 1):
        temp = deepcopy(prompt)
        temp.insert(i, newword)
        prompt_list.append(temp)
else:
    prompt_list.append(prompt)
    
generator = torch.Generator(device="cuda").manual_seed(238)

for eachprompt in prompt_list:
    p = ','.join(eachprompt)
    print(f'prompt: {p}')
    image = pipe(
        prompt = p,
        negative_prompt = negative_prompt,
        image = init_image,
        generator = generator,
        guidance_scale = 13.5,
        strength = 0.8,
        num_inference_steps = 50,
        num_images_per_prompt = 1).images[0]
    image.save(os.path.join('results', f'{p}.png'))

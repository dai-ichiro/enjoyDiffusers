from PIL import Image
import argparse
import glob
import os
import sys
import torch

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
parser.add_argument(
    '--mask',
    type=str,
    help='mask image'
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=1,
    help='how many samples to produce for each given prompt',
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

img_fname = opt.image

if opt.mask:
    mask_fname = opt.mask
else:
    mask_fname = os.path.splitext(os.path.basename(img_fname))[0] + '_mask.png'

if not os.path.exists(mask_fname):
    print('Unable to identify mask image')
    sys.exit()

init_image = Image.open(img_fname).convert('RGB').resize((512, 512))
mask_image = Image.open(mask_fname).convert('RGB').resize((512, 512))

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

prompt = 'good quality, best quality'
negatvie_prompt = 'poor quality, bad quality'

os.makedirs('results', exist_ok=True)

for i in range(opt.n_samples):
    image = pipe(
        prompt=prompt,
        negative_prompt = negatvie_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=50).images[0]

    image.save(os.path.join('results', f'result_{i+1}.png'))

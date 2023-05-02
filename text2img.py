import os
import argparse
import torch
from diffusers import DiffusionPipeline

def main(args):

    model_id = args.model_id
    scheduler = args.scheduler
    scale_list = args.scale
    steps = args.steps
    seed = args.seed

    if os.path.isfile(args.prompt):
        print(f'reading prompts from {args.prompt}')
        with open(args.prompt, 'r') as f:
            prompt = f.readlines()
            prompt = [x.strip() for x in prompt if x.strip() != '']
            prompt = ', '.join(prompt)
    else:
        prompt = '1girl, best quality, extremely detailed'

    if os.path.isfile(args.negative_prompt):
        print(f'reading negative prompts from {args.negative_prompt}')
        with open(args.negative_prompt, 'r') as f:
            negative_prompt = f.readlines()
            negative_prompt = [x.strip() for x in negative_prompt if x.strip() != '']
            negative_prompt = ','.join(negative_prompt)
    else:
        negative_prompt = None

    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        safety_checker=None,
        torch_dtype=torch.float16)
    
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

    pipe.to("cuda")

    os.makedirs('results', exist_ok=True)

    print(f'prompt: {prompt}')
    print(f'negative prompt: {negative_prompt}')
    
    scale_list = opt.scale
    steps = opt.steps

    for i in range(opt.n_samples):
        for scale in scale_list:
            seed  = opt.seed + i
            generator = torch.manual_seed(seed)
            image = pipe(
                prompt = prompt,
                negative_prompt = negative_prompt,
                generator = generator,
                guidance_scale = scale,
                num_inference_steps = steps,
                num_images_per_prompt = 1).images[0]
            image.save(os.path.join('results', f'{scheduler}_seed{seed}_scale{scale}_steps{steps}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        type=str,
        default='CompVis/stable-diffusion-v1-4',
        help='model id of the pipeline',
    )
    parser.add_argument(   
        '--prompt',
        type=str,
        default='prompt.txt',
        help='path to prompt file',
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default='negative_prompt.txt',
        help='path to negative_prompt file',
    )
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
        '--scheduler',
        type=str,
        default='pndm',
        choices=['pndm', 'multistepdpm', 'eulera']
    )
    opt = parser.parse_args()

    main(opt)

'''
python text2img.py ^
  --model model/Counterfeit-V2.0 ^
  --prompt prompt.txt ^
  --scheduler eulera ^
  --n_samples 10 
'''


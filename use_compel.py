from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from compel import Compel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--seed',
    type=int,
    default=20000,
    help='the seed (for reproducible sampling)',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=5,
    help='how many samples to produce for each given prompt',
)
parser.add_argument(
    '--steps',
    type=int,
    default=25,
    help='num_inference_steps',
)
args = parser.parse_args()

seed = args.seed
steps = args.steps
scale = 7.0

model_id = "./BRAV5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    torch_dtype=torch.float16)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.load_textual_inversion("embeddings", weight_name="EasyNegative.safetensors", token="EasyNegative")
pipe.load_textual_inversion("embeddings", weight_name="ng_deepnegative_v1_75t.pt", token="ng_deepnegative_v1_75t")
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
pipe.to("cuda")

prompt = "masterpiece+++, photorealistic+++, (best quality)+++, attractive++, (highly detailed)++, pretty Japanese woman, short hair, full body"
negative_prompt = "EasyNegative, ng_deepnegative_v1_75t, (Worst Quality)+++"

prompt_embeds = compel([prompt])
negative_prompt_embeds = compel([negative_prompt])

for i in range(args.n_samples):
    temp_seed = seed + i * 100
    generator = torch.Generator(device="cuda").manual_seed(temp_seed)
    image = pipe(
        prompt_embeds = prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=scale,
        width=768,
        height=1152,
        ).images[0]
    image.save(f"./step{steps}_seed{temp_seed}.png")
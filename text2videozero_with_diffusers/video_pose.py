from huggingface_hub import hf_hub_download

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

import imageio
from PIL import Image

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "model/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("controlnet/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
#latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16)

prompt = "Darth Vader dancing in a desert"

chunk_size = 2
frames = len(pose_images)
seed = 20000

import numpy as np
chunk_ids = np.arange(0, frames, chunk_size - 1)
result = []
for i in range(len(chunk_ids)):
    ch_start = chunk_ids[i]
    ch_end = ch_end = frames if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    frame_ids = [0] + list(range(ch_start, ch_end))
    ch_images = [pose_images[0]] + pose_images[ch_start:ch_end]
    print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
    #generator = torch.Generator('cuda').manual_seed(seed)
    generator = torch.manual_seed(seed)
    inference_result = pipe(
        prompt = [prompt]*len(frame_ids),
        image = ch_images,
        latents =latents.repeat(len(frame_ids), 1, 1, 1),
        generator = generator,
        output_type = 'numpy'
    ).images[1:]

    result.append(inference_result)

result_array = np.concatenate(result)

#result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result_array, fps=4)
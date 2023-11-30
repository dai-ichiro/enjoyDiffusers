import decord
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

video_path ="__assets__/poses_skeleton_gifs/dance1_corr.mp4"
vr = decord.VideoReader(video_path)
pose_images = [Image.fromarray(frame.asnumpy()).resize((1024, 1024)) for frame in vr]

pose_images = pose_images

controlnet_model_id = 'controlnet/controlnet-openpose-sdxl-1.0'
model_id = 'model/stable-diffusion-xl-base-1.0'

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to('cuda')

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

prompt = "woman wearing plain white t-shirt and blue denim, dancing in the forrest"
negative_prompt = "worst quality, low quality"

seed_1 = 10000
seed_2 = 1000000

# fix latents for all frames
torch.manual_seed(seed_1)
latents = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.float16)

chunk_size = 3
frames_count = len(pose_images)
chunk_ids = np.arange(0, frames_count, chunk_size - 1)

result = []
for i, ch_start in enumerate(chunk_ids):
    print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
    ch_end = frames_count if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    ids_count = (ch_end - ch_start) + 1
    ch_images = [pose_images[0]] + pose_images[ch_start:ch_end]
    generator=torch.manual_seed(seed_2)

    inference_result = pipe(
        prompt=[prompt] * ids_count,
        negative_prompt=[negative_prompt] * ids_count,
        image=ch_images,
        latents=latents.repeat(ids_count, 1, 1, 1),
        generator=generator,
        output_type = 'np'
        ).images[1:]
    result.append(inference_result)

result_array = np.concatenate(result)
result_array = [(r * 255).astype("uint8") for r in result_array]

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('video.mp4', fourcc, 2, (1024, 1024))

for frame in result_array:
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(img)

out.release()

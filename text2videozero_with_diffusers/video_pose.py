import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

video_path = 'dance1_corr.mp4'

## read video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

pose_images = []
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_images.append(Image.fromarray(frame))

model_id = "model/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("controlnet/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

chunk_size = 3
frames = len(pose_images)
seed = 20000

# fix latents for all frames
torch.manual_seed(seed)
latents = torch.randn((1, 4,  height//8, width//8), device="cuda", dtype=torch.float16)

prompt = "a beautiful girl dancing in a desert"

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

## make video
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))
for frame in result_array:
    frame = (frame * 255).astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
out.release()
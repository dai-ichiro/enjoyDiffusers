import decord
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

video_path ="__assets__/poses_skeleton_gifs/dance1_corr.mp4"
vr = decord.VideoReader(video_path)
pose_images = [Image.fromarray(frame.asnumpy()).resize((1024, 1024)) for frame in vr]

pose_images = pose_images[0:8]

controlnet_model_id = 'controlnet/controlnet-openpose-sdxl-1.0'
model_id = 'model/stable-diffusion-xl-base-1.0'

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_id,
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16
).to('cuda')

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
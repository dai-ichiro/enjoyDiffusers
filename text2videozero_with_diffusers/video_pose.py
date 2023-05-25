import os
import cv2
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

def main(args):
    video_path = args.video
    chunk_size = args.chunk_size
    seed = args.seed

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

    frames = len(pose_images)
    
    # fix latents for all frames
    torch.manual_seed(seed)
    latents = torch.randn((1, 4,  height//8, width//8), device="cuda", dtype=torch.float16)

    if args.prompt is not None and os.path.isfile(args.prompt):
        print(f'reading prompts from {args.prompt}')
        with open(args.prompt, 'r') as f:
            prompt_from_file = f.readlines()
            prompt_from_file = [x.strip() for x in prompt_from_file if x.strip() != '']
            prompt_from_file = ', '.join(prompt_from_file)
            prompt = f'{prompt_from_file}, best quality, extremely detailed'
    else:
        prompt = 'best quality, extremely detailed'
    
    negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    print(f'prompt: {prompt}')
    print(f'negative prompt: {negative_prompt}')

    chunk_ids = np.arange(0, frames, chunk_size - 1)
    result = []
    for i in range(len(chunk_ids)):
        ch_start = chunk_ids[i]
        ch_end = ch_end = frames if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
        frame_ids = [0] + list(range(ch_start, ch_end))
        ch_images = [pose_images[0]] + pose_images[ch_start:ch_end]
        print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
        generator = torch.manual_seed(seed)
        inference_result = pipe(
            prompt = [prompt]*len(frame_ids),
            negative_prompt = [negative_prompt]*len(frame_ids),
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--video',
        required=True,
        type=str,
        help='video path'
    )
    parser.add_argument(
        '--chunk_size',
        default=2,
        type=int,
        help='chunk_size'
    )
    parser.add_argument(
        '--seed',
        default=20000,
        type=int,
        help='seed'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='path to prompt file'
    )
    args = parser.parse_args()

    main(args)
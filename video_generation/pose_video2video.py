import os
import numpy as np
import torch
import cv2
import decord
from einops import rearrange
from controlnet_aux import OpenposeDetector

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL

from argparse import ArgumentParser

class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def main(args):
    video_path = args.video
    save_path = args.save_path
    model_id = args.model
    vae_folder = args.vae
    scheduler = args.scheduler
    seed = args.seed
    num_inference_steps = args.steps
    controlnet_conditioning_scale = args.conditioning_scale
    guidance_scale = args.guidance_scale
    eta=0.0

    if vae_folder is not None:
        vae = AutoencoderKL.from_pretrained('vae/anime2_vae', torch_dtype=torch.float16).to('cuda')
    else:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder='vae', torch_dtype=torch.float16).to('cuda')
    
    controlnet = ControlNetModel.from_pretrained('controlnet/sd-controlnet-openpose', torch_dtype=torch.float16).to('cuda')
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        vae = vae, 
        safety_checker=None,
        torch_dtype = torch.float16).to('cuda')

    controlnet_attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
    pipe.unet.set_attn_processor(processor=controlnet_attn_proc)
    pipe.controlnet.set_attn_processor(processor=controlnet_attn_proc)

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

    ## read video
    vr = decord.VideoReader(video_path)
    fps = int(vr.get_avg_fps())
    video = vr.get_batch(np.arange(len(vr))).asnumpy()

    ## detect pose
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    detected_maps = []
    for i, frame in enumerate(video):
        detected_map = openpose(frame)
        detected_array = np.array(detected_map)
        detected_maps.append(detected_array[None])

    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    control = rearrange(control, 'f h w c -> f c h w')

    frames_count, h, w, _ = video.shape
    generator = torch.Generator('cuda').manual_seed(seed)
    latents = torch.randn((1, 4, h//8, w//8), dtype=torch.float16, device='cuda', generator=generator)
    
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

    result = []
    latents = latents.repeat(2, 1, 1, 1)
    prompt = [prompt] * 2
    negative_prompt = [negative_prompt] *2
    for i in range(frames_count):
        print(f'{i+1}/{frames_count}')
        frame_ids = [0] + [i]
        
        image = control[frame_ids]
        
        generator = torch.Generator('cuda').manual_seed(seed)
        result.append(
            pipe(
                image = image,
                prompt = prompt,
                height=h,
                width=w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                eta=eta,
                negative_prompt = negative_prompt,
                latents = latents,
                generator = generator,
                output_type='numpy').images[1:])

    result = np.concatenate(result)
    
    ## make video
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in result:
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
        '--save_path',
        default='canny_result.mp4',
        type=str,
        help='save path'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='model',
    )   
    parser.add_argument(
        '--vae',
        type=str,
        help='vae'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='pndm',
        choices=['pndm', 'multistepdpm', 'eulera']
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='num_inference_steps'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='seed'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        help='prompt'
    )
    parser.add_argument(
        '--conditioning_scale',
        type=float,
        default=1.0,
        help='conditioning_scale'
    )
    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=9.0,
        help='guidance_scale'
    )
    args = parser.parse_args()

    main(args)



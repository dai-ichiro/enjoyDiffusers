import numpy as np
import torch
import cv2
import decord
from einops import rearrange

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

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

def main(
    video_path,
    prompt,
    save_path,                            
    num_inference_steps=20,
    controlnet_conditioning_scale=1.0,
    guidance_scale=9.0,
    seed=42,
    eta=0.0,
    low_threshold=100,
    high_threshold=200):

    vae = AutoencoderKL.from_pretrained('vae/anime2_vae', torch_dtype=torch.float16).to('cuda')
    
    controlnet = ControlNetModel.from_pretrained('controlnet/sd-controlnet-canny', torch_dtype=torch.float16).to('cuda')
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'local_model/anything-v4.0',
        controlnet=controlnet,
        vae = vae, 
        safety_checker=None,
        torch_dtype = torch.float16).to('cuda')

    controlnet_attn_proc = CrossFrameAttnProcessor(unet_chunk_size=2)
    pipe.unet.set_attn_processor(processor=controlnet_attn_proc)
    pipe.controlnet.set_attn_processor(processor=controlnet_attn_proc)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    added_prompt = 'best quality, extremely detailed'
    negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    
    ## read video
    vr = decord.VideoReader(video_path)
    fps = int(vr.get_avg_fps())
    video = vr.get_batch(np.arange(len(vr))).asnumpy()

    ## detect edge
    detected_maps = []
    for frame in video:
        detected_map = cv2.Canny(frame, low_threshold, high_threshold)
        detected_map = np.stack((detected_map, detected_map, detected_map), axis=2)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    control = rearrange(control, 'f h w c -> f c h w')

    f, h, w, _ = video.shape
    generator = torch.Generator('cuda').manual_seed(seed)
    latents = torch.randn((1, 4, h//8, w//8), dtype=torch.float16, device='cuda', generator=generator)
    
    seed = seed
    prompt='a beautiful girl running'
    prompt = prompt + ', ' + added_prompt
    negative_prompt=negative_prompts

    result = []
    latents = latents.repeat(2, 1, 1, 1)
    prompt = [prompt] * 2
    negative_prompt = [negative_prompt] *2
    for i in range(f):
        print(f'{i+1}/{f}')
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
        '--low_threshold',
        type=int,
        default=50,
        help='low_threshold'
    )
    parser.add_argument(
        '--high_threshold',
        type=int,
        default=50,
        help='high_threshold'
    )
    args = parser.parse_args()

    main(
        prompt='a beautiful girl running',
        video_path = args.video,
        low_threshold = args.low_threshold,
        high_threshold = args.high_threshold,
        save_path = args.save_path
    )



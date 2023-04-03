import numpy as np
import torch

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler

import myutils
import os
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"


class Model:
    def __init__(self, **kwargs):
        self.device = 'cuda'
        self.dtype = torch.float16
        self.generator = torch.Generator('cuda')

    def inference_chunk(self, frame_ids, **kwargs):
        if self.pipe is None:
            return

        prompt = np.array(kwargs.pop('prompt'))
        negative_prompt = np.array(kwargs.pop('negative_prompt', ''))
        latents = None
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if 'image' in kwargs:
            kwargs['image'] = kwargs['image'][frame_ids]
        if 'video_length' in kwargs:
            kwargs['video_length'] = len(frame_ids)
        return self.pipe(prompt=prompt[frame_ids].tolist(),
                         negative_prompt=negative_prompt[frame_ids].tolist(),
                         latents=latents,
                         generator=self.generator,
                         **kwargs)

    def inference(self, split_to_chunks=False, chunk_size=8, **kwargs):
        if self.pipe is None:
            return
        seed = kwargs.pop('seed', 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop('generator', '')

        if 'image' in kwargs:
            f = kwargs['image'].shape[0]
        else:
            f = kwargs['video_length']

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)
            result = []
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                self.generator.manual_seed(seed)
                print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
                result.append(self.inference_chunk(frame_ids=frame_ids,
                                                   prompt=prompt,
                                                   negative_prompt=negative_prompt,
                                                   **kwargs).images[1:])
                frames_counter += len(chunk_ids)-1
                if on_huggingspace and frames_counter >= 80:
                    break
            result = np.concatenate(result)
            return result
        else:
            return self.pipe(prompt=prompt, negative_prompt=negative_prompt, generator=self.generator, **kwargs).images

    def process_controlnet_canny(self,
                                 video_path,
                                 prompt,
                                 chunk_size=8,
                                 num_inference_steps=20,
                                 controlnet_conditioning_scale=1.0,
                                 guidance_scale=9.0,
                                 seed=42,
                                 eta=0.0,
                                 low_threshold=100,
                                 high_threshold=200,
                                 resolution=512,
                                 save_path=None):

        controlnet = ControlNetModel.from_pretrained("controlnet/sd-controlnet-canny", torch_dtype=torch.float16).to('cuda')
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "local_model/anything-v4.0", 
            safety_checker=None,
            torch_dtype = torch.float16,
            controlnet=controlnet).to(self.device)
        
        #self.set_model(model_id="local_model/anything-v4.0", controlnet=controlnet)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        controlnet_attn_proc = myutils.CrossFrameAttnProcessor(unet_chunk_size=2)
        self.pipe.unet.set_attn_processor(processor=controlnet_attn_proc)
        self.pipe.controlnet.set_attn_processor(processor=controlnet_attn_proc)

        added_prompt = 'best quality, extremely detailed'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        video, fps = myutils.prepare_video(
            video_path, resolution, self.device, self.dtype, False)
        control = myutils.pre_process_canny(
            video, low_threshold, high_threshold).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return myutils.create_video(result, fps, path=save_path)

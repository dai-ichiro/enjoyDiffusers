import torch
from diffusers import QwenImage21Pipeline, FlowMatchEulerDiscreteScheduler

pipe = QwenImage21Pipeline.from_pretrained(
    "models/Qwen-Image-2.1",
    dtype=torch.bfloat16
)

pipe.transformer.load_lora_adapter(
    "models/Qwen-Image-2.1-viggle-turbo",
    subfolder="peft_v0.2.1",
    weight_name="adapter_model.safetensors",
    prefix=None
)

pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "Viggle/Qwen-Image-2.1-viggle-turbo",
    subfolder="scheduler"
)

pipe.enable_model_cpu_offload()

 # the v0.2.1 schedule; pass both to every call
STEPS, SIGMAS = 6, [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]

image = pipe(
    prompt="A studio portrait of an old fisherman mending a net, warm rim light, 85mm.",
    height=1024,
    width=1024,
    num_inference_steps=STEPS,
    sigmas=SIGMAS,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

image.save("out.png")
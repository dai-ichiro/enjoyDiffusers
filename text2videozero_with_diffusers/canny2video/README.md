
## Environment

~~~
Ubuntu22.04 on WSL2
CUDA 11.8
Python 3.10
~~~

## Requirements

~~~
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors opencv-python
pip install decord einops
~~~

## How to use

~~~
python canny_video2video.py \
  --video run640.mp4 \
  --model model/anything-v4.0 \
  --vae vae/anime2_vae \
  --scheduler eulera \
  --low_threshold 50 \
  --high_threshold 100 \
  --prompt prompt.txt \
  --seed 20000 \
  --steps 30 \
  --save_path run_output.mp4
~~~

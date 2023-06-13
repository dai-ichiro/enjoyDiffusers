
## Environment

~~~
Ubuntu22.04 on WSL2
CUDA 11.8
Python 3.10
~~~

## Requirements

~~~
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/diffusers.git
pip install transformers accelerate safetensors opencv-python
~~~

## How to use

「dance1_corr.mp4」was downloaded from [here](https://huggingface.co/spaces/PAIR/Text2Video-Zero/blob/main/__assets__/poses_skeleton_gifs/dance1_corr.mp4).
~~~
python video_pose.py \
  --video dance1_corr.mp4 \
  --chunk_size 3 \
  --prompt prompt.txt \
  --seed 20000
~~~

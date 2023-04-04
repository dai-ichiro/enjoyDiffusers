# Text2Video-Zero with ControlNet

## Requirements

~~~
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r https://raw.githubusercontent.com/Picsart-AI-Research/Text2Video-Zero/main/requirements.txt
~~~

## How to use

~~~
python canny_video2video.py ^
  --video run.mp4 ^
  --model model/anything-v4.0 ^
  --vae vae/anime2_vae ^
  --scheduler eulera ^
  --low_threshold 50 ^
  --high_threshold 50 ^
  --prompt prompt.txt ^
  --seed 42 ^
  --steps 30
~~~

## Link to my blog

https://touch-sp.hatenablog.com/entry/2023/04/01/225144

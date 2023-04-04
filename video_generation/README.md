# Text2Video-Zero with ControlNet

## How to use

~~~
python canny_video2video.py ^
  --video run.mp4 ^
  --model model/anything-v4.0 ^
  --vae vae/anime2_vae ^
  --sheduler eulera ^
  --low_threshold 50 ^
  --high_threshold 50 ^
  --prompt prompt.txt ^
  --seed 42
  --steps 30
~~~

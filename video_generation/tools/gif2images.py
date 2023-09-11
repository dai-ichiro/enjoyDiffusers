from PIL import Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gif',
    type=str,
    help='gif file'
)
opt = parser.parse_args()

gif_images = Image.open(opt.gif)

for i in range(gif_images.n_frames):
    gif_images.seek(i)
    gif_images.save(f"{i}.png")



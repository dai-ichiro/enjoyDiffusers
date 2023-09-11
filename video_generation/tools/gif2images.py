from PIL import Image
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gif',
    type=str,
    help='gif file'
)
opt = parser.parse_args()

gif_path = opt.gif

gif_images = Image.open(gif_path)

save_folder = os.path.splitext(os.path.basename(gif_path))[0]
os.makedirs(save_folder, exist_ok=True)

for i in range(gif_images.n_frames):
    gif_images.seek(i)
    gif_images.save(os.path.join(save_folder, f"{i}.png"))



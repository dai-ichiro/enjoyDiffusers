from diffusers.utils import export_to_gif, load_image
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder',
    type=str,
    help='folder path'
)
opt = parser.parse_args()

images = glob.glob(f"{opt.folder}/*.png")

image_list = []
for i in range(len(images)):
    image_list.append(load_image(f"{opt.folder}/{i}.png")
                    )
export_to_gif(image_list, "result.gif")
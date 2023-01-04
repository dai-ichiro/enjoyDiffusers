from PIL import Image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='path to RGBA image'
)
parser.add_argument(
    '--ext',
    type=str,
    default='jpg',
    help='extension'
)
opt = parser.parse_args()

img_path = opt.image
img_fname = os.path.splitext(os.path.basename(img_path))[0]

img = Image.open(img_path)
new = Image.new('RGB', img.size, (255, 255, 255))
new.paste(img, mask=img.split()[3])
new.save(f'rgb_{img_fname}.{opt.ext}')
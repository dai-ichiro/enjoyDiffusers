import numpy as np
from PIL import Image
import argparse

# アスペクト比を固定して、幅が指定した値になるようリサイズする。
def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))

# アスペクト比を固定して、高さが指定した値になるようリサイズする。
def scale_to_height(img, height):
    width = round(img.width * height / img.height)
    return img.resize((width, height))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--list',
    required=True,
    nargs="*",
    type=str,
    help='image list')

parser.add_argument(
    '--vertical',
    action='store_true',
    help='vertical')
args = parser.parse_args()

image_list =[Image.open(x) for x in args.list]

if args.vertical:
    width = image_list[0].width
    image_list = [scale_to_width(x, width) for x in image_list]
    array_list = [np.array(x) for x in image_list]
    one_array = np.vstack(tuple(array_list))
else:
    height = image_list[0].height
    image_list = [scale_to_height(x, height) for x in image_list]
    array_list = [np.array(x) for x in image_list]
    one_array = np.hstack(tuple(array_list))
    
pil_image = Image.fromarray(one_array)
pil_image.save('stackimage.png')
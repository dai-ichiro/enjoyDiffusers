import cv2
from rembg import remove, new_session

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type=str,
    help='original video'
)
parser.add_argument(
    '--type',
    type=str,
    default="u2net_human_seg",
    choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use"],
    help='rembg model'
)
opt = parser.parse_args()

image_path = opt.input

im = cv2.imread(image_path)

im = remove(im, session=new_session(opt.type))

cv2.imwrite("result.png", im)





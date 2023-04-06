import os
import cv2
from decord import VideoReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True, help='original video')
parser.add_argument('--x_min', type=int, required=True, help='x_min')
parser.add_argument('--width', type=int, required=True, help='width')
parser.add_argument('--y_min', type=int, required=True, help='y_min')
parser.add_argument('--height', type=int, required=True, help='height')
opt = parser.parse_args()

x_min = opt.x_min
x_max = x_min + opt.width

y_min = opt.y_min
y_max = y_min + opt.height

vr = VideoReader(opt.video)

os.makedirs('images',exist_ok=True)

for i, image in enumerate(vr):
    numpy_array = image.asnumpy()
    numpy_array = numpy_array[y_min:y_max, x_min:x_max, :]
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('images', f'{i}.png'), numpy_array)

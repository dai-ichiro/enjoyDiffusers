import os
import cv2
from decord import VideoReader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--video',
    type=str,
    help='original video'
)
opt = parser.parse_args()

vr = VideoReader(opt.video)

os.makedirs('images',exist_ok=True)

for i, image in enumerate(vr):
    numpy_array = image.asnumpy()
    numpy_array = numpy_array[0:512, 224:(224+512), :]
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join('images', f'{i}.png'), numpy_array)

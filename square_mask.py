import os
import cv2
import numpy as np
from copy import deepcopy

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument(
    '--image', 
    type=str, 
    required=True,
    help='original image' )
opt = parser.parse_args()

img_path = opt.image
img_fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]

img = cv2.imread(img_path)

temp_img = deepcopy(img)
width, height = temp_img.shape[0:2]
maskimg = np.zeros(temp_img.shape[0:2])

position_x = 100
position_y = 100

source_window = "make_mask"
cv2.namedWindow(source_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(source_window, width, height)
cv2.moveWindow(source_window, position_x, position_y)

while True:
    rect = cv2.selectROI(source_window, temp_img, False, False)
    
    if rect == (0, 0, 0, 0):
        break
    else:
        xmin, ymin, width, height = rect
        xmax = xmin + width
        ymax = ymin + height

        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
        cv2.rectangle(temp_img, (xmin,ymin), (xmax, ymax), (255, 0, 0), 5)
        cv2.rectangle(maskimg, (xmin,ymin), (xmax, ymax), 255, -1)

cv2.destroyAllWindows()
cv2.imwrite(f'{img_fname_no_ext}_mask.png', maskimg)


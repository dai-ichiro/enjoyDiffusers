import os
import cv2
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--image', type=str, help='original image' )
opt = parser.parse_args()

img_path = opt.image
img_fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]

img = cv2.imread(img_path)

source_window = "make_mask"
cv2.namedWindow(source_window)

rect = cv2.selectROI(source_window, img, False, False)
cv2.destroyAllWindows()

xmin, ymin, width, height = rect

maskimg = np.zeros(img.shape)

cv2.rectangle(maskimg, (xmin, ymin), (xmin+width, ymin+height), (255,255,255), -1)

cv2.imwrite(f'{img_fname_no_ext}_mask.png', maskimg)


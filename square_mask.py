import cv2
import numpy as np

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--image', type=str, help='original image' )
args = parser.parse_args()

original_image = args.image

img = cv2.imread(original_image)

source_window = "make_mask"
cv2.namedWindow(source_window)

rect = cv2.selectROI(source_window, img, False, False)
cv2.destroyAllWindows()

xmin, ymin, width, height = rect

maskimg = np.zeros(img.shape)

cv2.rectangle(maskimg, (xmin, ymin), (xmin+width, ymin+height), (255,255,255), -1)

cv2.imwrite('mask.png', maskimg)


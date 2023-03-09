import os
import cv2
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    type=str,
    help='original image'
)
parser.add_argument(
    '--size',
    type=int,
    default=20,
    help='rectangle size'
)

opt = parser.parse_args()
size = opt.size

img_path = opt.image
img_fname_no_ext = os.path.splitext(os.path.basename(img_path))[0]

drawing = False 

def draw_circle(event,x,y, flags, param):

    global drawing, mask_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing == True:
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
            xmin = x - size
            ymin = y - size
            xmax = x + size
            ymax = y + size

            cv2.rectangle(original_image, (xmin,ymin), (xmax, ymax), (255, 255, 255), -1)
            cv2.rectangle(mask_image, (xmin,ymin), (xmax, ymax), 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
original_image = cv2.imread(img_path)
mask_image = np.zeros(original_image.shape[0:2])

width, height = original_image.shape[0:2]
position_x = 100
position_y = 100

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', width, height)
cv2.moveWindow('image', position_x, position_y)
cv2.setMouseCallback('image', draw_circle)

while True:  
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'{img_fname_no_ext}_mask.png', mask_image)
    elif key == 27:
        break
    elif key == ord('u'):
        position_y -= 200
        cv2.moveWindow('image', position_x, position_y)
    elif key == ord('d'):
        position_y += 200
        cv2.moveWindow('image', position_x, position_y)
    cv2.imshow('image',original_image)
    
cv2.destroyAllWindows()
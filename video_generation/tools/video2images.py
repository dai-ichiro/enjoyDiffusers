import cv2
import os 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--video',
    type=str,
    help='original video'
)
opt = parser.parse_args()

os.makedirs('images', exist_ok=True)

cap = cv2.VideoCapture(opt.video)

frame_no = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    cv2.imwrite(os.path.join('images', f'{frame_no}.png'), frame)
    frame_no += 1

cap.release()





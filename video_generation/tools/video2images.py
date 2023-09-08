import cv2
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--video',
    type=str,
    help='original video'
)
parser.add_argument(
    '--rembg',
    type=str,
    choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use"],
    help='rembg model'
)
opt = parser.parse_args()

video_path = opt.video
video_fname = os.path.splitext(os.path.basename(video_path))[0]

if opt.rembg:
    from rembg import remove, new_session
    save_folder = f'{opt.rembg}_{video_fname}'
else:
    save_folder = video_fname
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_no = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        break
    
    if opt.rembg:
        frame = remove(frame, session=new_session(opt.rembg))

    cv2.imwrite(os.path.join(save_folder, f'{frame_no}.png'), frame)
    frame_no += 1

cap.release()





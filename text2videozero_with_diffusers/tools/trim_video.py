import cv2
import os

pose_folder = 'pose'

fps = 4
width = 512
height = 480
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('resized_video.mp4', fourcc, fps, (width, height))

for i in range(16):
    image1 = cv2.imread(os.path.join(pose_folder, f'{i}.png'))
    trim_image = image1[0:480, :, :]
    out.write(trim_image)

out.release()
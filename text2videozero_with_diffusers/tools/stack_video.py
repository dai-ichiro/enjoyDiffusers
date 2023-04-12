import cv2
import os
import numpy as np

image_folder = 'result'
pose_folder = 'pose'

fps = 4
width = 1025
height = 512
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

for i in range(16):
    image1 = cv2.imread(os.path.join(pose_folder, f'{i}.png'))
    image2 = cv2.imread(os.path.join(image_folder, f'{i}.png'))
    result = np.hstack((image1, image2))
    out.write(result)

out.release()






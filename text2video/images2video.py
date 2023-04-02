import cv2
import os
import glob

images_list = glob.glob('images/*.png')

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4',fourcc, 4, (512, 512))

for i in range(len(images_list)):
    image = cv2.imread(os.path.join('images', f'{i}.png'))
    out.write(image)

out.release()

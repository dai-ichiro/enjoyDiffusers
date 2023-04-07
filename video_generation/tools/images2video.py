import cv2
import glob
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder',
    type=str,
    help='folder path'
)
parser.add_argument(
    '--fps',
    type=int,
    default = 8,
    help='frame per second'
)
opt = parser.parse_args()

images_list = glob.glob(f'{opt.folder}/*.png')

h, w, _ = (cv2.imread(images_list[0])).shape

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4',fourcc, opt.fps, (w, h))

for i in range(len(images_list)):
    image = cv2.imread(os.path.join(opt.folder, f'{i}.png'))
    out.write(image)

out.release()

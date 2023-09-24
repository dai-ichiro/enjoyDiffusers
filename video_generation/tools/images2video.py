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
parser.add_argument(
    '--start',
    type=int,
    default = 0,
    help='number of first frame'
)
parser.add_argument(
    '--length',
    type=int,
    help='frame count'
)
opt = parser.parse_args()

images_list = glob.glob(f'{opt.folder}/*.png')

h, w, _ = (cv2.imread(images_list[0])).shape

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4',fourcc, opt.fps, (w, h))

start_frame = opt.start
if opt.length is None:
    length = len(images_list)
else:
    length = opt.length

for i in range(length):
    #image = cv2.imread(os.path.join(opt.folder, f'{i}.png'))
    image = cv2.imread(os.path.join(opt.folder, f'{i+start_frame}.png'))
    out.write(image)

out.release()

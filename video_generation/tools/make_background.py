import os
import cv2
import numpy as np
from copy import deepcopy

save_dir = 'bg_images'
os.makedirs(save_dir, exist_ok=True)

original_image = 'background.png'

original_im = cv2.imread(original_image)

images_count = 47
steps = 6

assert (images_count * steps) % 2 ==0, 'error'
trim_width = int((images_count * steps) / 2)

trim_im = original_im[:, 0:trim_width, :]
reverse_im = trim_im[:, ::-1, :]

bg_im = deepcopy(trim_im)
while True:
    bg_im = np.hstack((bg_im, reverse_im))
    
    if bg_im.shape[1] > (images_count * steps + 672):
        break
    
    bg_im = np.hstack((bg_im, trim_im))

    if bg_im.shape[1] > (images_count * steps + 672):
        break

for i in range(images_count + 1):
    trim_bg = bg_im[:, (i * steps):(i * steps) + 672, :]
    cv2.imwrite(os.path.join(save_dir, f'{i}.png'), trim_bg)

import os
import cv2
import numpy as np

save_dir = 'bg_images'
os.makedirs(save_dir, exist_ok=True)

original_iamge = 'result.png'
original_width = 1024
original_height = 512

image_num = 47

steps = 40

end = steps * (image_num - 1) + 672

trim_size = int(end / 2)

ratio = original_width / original_height
original_im = cv2.imread(original_iamge)
original_im = cv2.resize(original_im, (int(672 * ratio), 672))

trim_im = original_im[:, 0:trim_size, :]
reverse_im = trim_im[:, ::-1, :]

result_im = np.hstack((trim_im, reverse_im))

print(result_im.shape)

for i in range(image_num):
    im = result_im[:, (i * steps):(i * steps + 672), :]
    cv2.imwrite(os.path.join(save_dir, f'{i}.png'), im)
import os
import glob
from PIL import Image

original_image_folder = 'u2net_human'
bg_image_folder = 'bg_images'

save_dir = 'bg_merge'
os.makedirs(save_dir, exist_ok=True)

original_image_count = len(glob.glob(os.path.join(original_image_folder, '*.png')))
bg_image_count = len(glob.glob(os.path.join(bg_image_folder, '*.png')))

assert original_image_count==bg_image_count, 'image count mismatch'

for i in range(original_image_count):

    original_im = Image.open(os.path.join(original_image_folder, f'{i}.png'))

    bg_im = Image.open(os.path.join(bg_image_folder, f'{i}.png'))
    bg_im.putalpha(alpha=255)

    out = Image.alpha_composite(bg_im, original_im)

    out.save(os.path.join(save_dir, f'{i}.png'))

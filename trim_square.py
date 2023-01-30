import warnings
warnings.simplefilter('ignore', UserWarning)

import os
import cv2
import torch
import mmcv
from mmtrack.apis import inference_sot, init_model
from mim.commands.download import download

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--video', type=str, default='video', help='video file name')
parser.add_argument('--skip', type=int, default=1, help='skip')
parser.add_argument('--mergin', type=int, default=0, help='mergin')
args = parser.parse_args()

video_fname = args.video
save_fname = os.path.splitext(os.path.basename(video_fname))[0]
skip = args.skip
mergin = args.mergin

def tracking():
    
    os.makedirs(save_fname, exist_ok=True)

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)
    checkpoint_name = 'siamese_rpn_r50_20e_lasot'
    checkpoint = download(package='mmtrack', configs=[checkpoint_name], dest_root="models")[0]
    model = init_model(os.path.join('models', checkpoint_name + '.py'), os.path.join('models', checkpoint), device=device)

    # tracking
    frames = mmcv.VideoReader(video_fname)
    
    h = frames.height
    w = frames.width

    source_window = "draw_rectangle"
    cv2.namedWindow(source_window)
    rect = cv2.selectROI(source_window, frames[0], False, False)
    # rect:(x1, y1, w, h)
    # convert (x1, y1, w, h) to (x1, y1, x2, y2)
    rect_convert = (rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3])
    cv2.destroyAllWindows()

    for frame_index, frame in enumerate(frames):
        result = inference_sot(model, frame, rect_convert, frame_id=frame_index)
        if frame_index % skip == 0:
            bbox = result['track_bboxes']
            # bbox:(x1, y1, x2, y2)
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            rect_width = bbox[2] - bbox[0]
            rect_height = bbox[3] - bbox[1]
            square_width = max(rect_width, rect_height)
            square_width_half = int(square_width / 2)

            new_x1 = center_x - square_width_half - mergin
            new_x2 = center_x + square_width_half + mergin
            new_y1 = center_y - square_width_half - mergin
            new_y2 = center_y + square_width_half + mergin

            if new_x1 >= 0 and new_x2 <= w and new_y1 >= 0 and new_y2 <= h:

                filename = f'{save_fname}_{frame_index}.png'
                trim_image = frame[new_y1:new_y2, new_x1:new_x2, :]
                cv2.imwrite(os.path.join(save_fname, filename), trim_image)    

if __name__ == '__main__':
    tracking()
    

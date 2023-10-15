from models import TRTModule  # isort:skip
import argparse, time, mss, numpy as np, pygetwindow as gw
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list



CLASSES   = ['bo', 'bo_h', 'fbo', 'fbo_h', 'be', 'be_h', 'fbe', 'fbe_h', 't', 't_h', 'dbo', 'dbe', 'dfbo', 'dfbe', 'dt' ]
cpy = COLORS.copy()
n = 0
for key, value in cpy.items():
    if n < len(CLASSES):
        COLORS[CLASSES[n]] = value
        n+=1

def get_winRT_win():
    wins = gw.getAllWindows()
    rt_win  = None
    for w in wins:
        if " - " in w.title and 'RT' in w.title and not "Tensor" in w.title:
            rt_win = w
            break
    if not rt_win:                
        print("window not found, start first and then restart this\nPress enter to exit")
        return                 
        input()
        sys.exit()
    return rt_win
sct = mss.mss()
w = get_winRT_win()

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    sct = mss.mss()

    while 1:

        x_of = int(7 + (1920 - 1080)/2); 
        y_of = 30;
        bounding_box = {'top': w.top + y_of, 'left': w.left + x_of, 'width': 1080  , 'height':1080}
        last_time = time.time()
        sct_img = sct.grab(bounding_box)
        scr_img = np.array(sct_img)

        scr_img_ = scr_img [:,:,:3]
        #save_image = save_path / image.name
        bgr = scr_img_#cv2.imread(str(image))
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)
        # inference
        last_time = time.time()

        data = Engine(tensor)
        print(f"fps: {round(1 / (time.time() - last_time), 2):5} {round(time.time() - last_time, 5):8}")

        bboxes, scores, labels = det_postprocess(data)
        # if bboxes.numel() == 0:
        #     # if no bounding box
        #     print(f': no object!')
        #     continue
        bboxes -= dwdh
        bboxes /= ratio
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(1 )
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

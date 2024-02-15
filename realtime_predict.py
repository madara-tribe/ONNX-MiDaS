import argparse
import sys
import numpy as np
import cv2
import torch

from utils import normalize_depth, download_mids, download_img


def get_cap(opt):
    if opt.video_path:
        cap = cv2.VideoCapture(opt.video_path)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('can not open video')
        exit()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))  # FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_frame = cv2.VideoWriter(opt.output_file, fourcc, fps, (frame_width, frame_height), isColor=False)
    return cap, out_frame
    
  
def midas_prediction(frame, device, midas, transform):
    input_batch = transform(frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_frame = prediction.cpu().numpy()
    #depth_frame = normalize_depth(depth_frame, bits=2)
    depth_frame = cv2.normalize(depth_frame, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_frame = (depth_frame*255).astype(np.uint8)
    depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_MAGMA)
    return depth_frame
    
    
def main(opt):
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas, transform = download_mids(model_type=opt.mode)
    cap, out_format = get_cap(opt)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        colored_depth = midas_prediction(frame, device, midas, transform)
        if opt.show:
            cv2.imshow('Depth Frame', colored_depth)
            print('now stream saveing')
        #out_format.write(colored_depth)
        cv2.imwrite('output/frame{}.png'.format(i), colored_depth)
        i += 1
    cap.release()
    out_format.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='DPT_Large', help='DPT_Large / DPT_Hybrid/ MiDaS_small')
    parser.add_argument('-o', '--output_file', type=str, default='output/driving.mp4', help='movie output path')
    parser.add_argument('-v', '--video_path', type=str, default='input/driving.mov', help='movie path for stream prediction')
    parser.add_argument('-s', '--show', action='store_true', help='prepare test data')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise



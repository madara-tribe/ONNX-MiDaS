import argparse
import sys
import cv2
import torch

from utils import normalize_depth, download_mids, download_img


def optput_format(opt, cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))  # FPS

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_format = cv2.VideoWriter(opt.video_output_path, fourcc, fps, (frame_width, frame_height), False)
    return out_format



def main(opt):
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas, transform = download_mids(model_type=opt.mode)
    if opt.video_path:
        cap = cv2.VideoCapture(opt.video_path)
        if not cap.isOpened():
            print('cat not open video')
            exit()
    else:
        cap = cv2.VideoCapture(0)
    out_format = optput_format(opt, cap)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        depth_frame = normalize_depth(depth_frame, bits=2)
        #if opt.show:
            #cv2.imshow('Depth Frame', depth_frame)
        out_format.write(depth_frame)
        print('now stream saveing')
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out_format.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='DPT_Large', help='DPT_Large / DPT_Hybrid/ MiDaS_small')
    parser.add_argument('-o', '--video_output_path', type=str, default='output/driving.mp4"', help='movie output path')
    parser.add_argument('-v', '--video_path', type=str, default='input/driving.mov', help='movie path for stream prediction')

    parser.add_argument('-s', '--show', action='store_true', help='prepare test data')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise

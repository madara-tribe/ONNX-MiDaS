import argparse
import sys
import numpy as np
import cv2
import torch
import onnxruntime as rt

from utils import download_mids, call_transform

def get_cap(opt, onnx_mode=False):
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
    if onnx_mode:
        out_frame = cv2.VideoWriter(opt.output_file+"_onnx.mp4", fourcc, fps, (frame_width, frame_height), isColor=False)
    else:
        out_frame = cv2.VideoWriter(opt.output_file+".mp4", fourcc, fps, (frame_width, frame_height), isColor=False)
    return cap, out_frame, frame_width, frame_height
    
  
def onnx_prediction(frame, transform, onnx_model, net_h, net_w):
    img_input = transform({"image": frame})["image"]
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    onnx_output = onnx_model.run([output_name], {input_name: img_input.reshape(1, 3, net_h, net_w).astype(np.float32)})[0]
    depth_frame = cv2.normalize(onnx_output[0], None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_frame = (depth_frame*255).astype(np.uint8)
    return cv2.applyColorMap(depth_frame, cv2.COLORMAP_INFERNO)
    

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
    depth_frame = cv2.normalize(depth_frame, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_frame = (depth_frame*255).astype(np.uint8)
    depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_INFERNO)
    return depth_frame
    
    
def main(opt):
    device = torch.device("cpu") #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if 'onnx' in opt.onnx_model_path:
        print("start onnx stream prediction")
        onnx_model = rt.InferenceSession(opt.onnx_model_path)
        transform, net_h, net_w = call_transform()
        mode_onnx = True
    else:
        print("start pytorch midas stream prediction")
        midas, transform = download_mids(model_type=opt.mode)
        mode_onnx = False
    cap, out_format, frame_width, frame_height = get_cap(opt, onnx_mode=mode_onnx)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if mode_onnx:
            colored_depth = onnx_prediction(frame, transform, onnx_model, net_h, net_w)
            colored_depth = cv2.resize(colored_depth, (frame_width, frame_height))
        else:
            colored_depth = midas_prediction(frame, device, midas, transform)
        if opt.show:
            cv2.imshow('Depth Frame', colored_depth)
            print('now stream saveing')
        
        colored_depth = cv2.cvtColor(colored_depth, cv2.COLOR_BGR2GRAY)
        out_format.write(colored_depth)
        #cv2.imwrite('output/frame{}.png'.format(i), colored_depth)
        i += 1
    cap.release()
    out_format.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='DPT_Large', help='DPT_Large / DPT_Hybrid/ MiDaS_small')
    parser.add_argument('-o', '--output_file', type=str, default='output/movie', help='movie output path')
    parser.add_argument('-v', '--video_path', type=str, default='input/movie.mp4', help='movie path for stream prediction')
    parser.add_argument('-onnx', '--onnx_model_path', type=str, default='weights/model-f6b98070.onnx', help='onnx midas weight model')
    parser.add_argument('-s', '--show', action='store_true', help='prepare test data')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise

import argparse
import sys
import time
import onnxruntime as rt
import cv2
import torch
import os
import numpy as np

from utils import read_image, call_transform

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def inference(opt):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    transform, net_h, net_w = call_transform()
    image = read_image(opt.filename)
    
    img_input = transform({"image": image})["image"]
    print('start prediction')
    start_time = time.time()
    onnx_model = rt.InferenceSession(opt.onnx_model_path)
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name
    onnx_output = onnx_model.run([output_name], {input_name: img_input.reshape(1, 3, net_h, net_w).astype(np.float32)})[0]
    print(onnx_output.shape)
    depth_img = onnx_output[0]
    output_norm = cv2.normalize(depth_img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #output_norm = normalize_depth(output_norm, bits=2)
    colored_depth = (output_norm*255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_MAGMA)
    print("Prediction took {:.2f} seconds".format(time.time() - start_time))
    cv2.imwrite(opt.out_name + '.jpg', output_norm)
    cv2.imwrite(opt.out_name + '_colored.jpg', colored_depth)
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename', type=str, default='input/input.jpg', help='input image name')
    parser.add_argument('-o', '--out_name', type=str, default='output/result', help='optput image name, must png not jpg')
    opt = parser.parse_args()
    try:
        inference(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise
"""


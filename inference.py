import time
import cv2

import torch
import numpy as np
import onnx
#from utils import read_image, call_transform
from onnxsim import simplify
import onnxruntime as rt
from midas.model_loader import load_model 

def draw_depth(depth_map, img_width, img_height):
    min_depth = np.inf
    max_depth = -np.inf
    # Normalize estimated depth to color it
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    min_depth = min_depth if min_depth < min_depth else min_depth
    max_depth = max_depth if max_depth > max_depth else max_depth

    print(min_depth, max_depth)
    norm_depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map = 255 - norm_depth_map

    # Normalize and color the image
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1),
                                    cv2.COLORMAP_JET)

    # Resize the depth map to match the input image shape
    return cv2.resize(color_depth, (img_width, img_height))
        
def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img
def prepare_input(path, input_width, input_height):
    img = cv2.imread(path)
    img_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_input, (input_width, input_height),
                           interpolation=cv2.INTER_AREA) / 255.0
    img_input = img_input.transpose(2, 0, 1)
    return img_input[np.newaxis, :, :, :].astype(np.float32)
        
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_, transform, net_w, net_h = load_model(device, model_path='weights/dpt_large_384.pt', model_type="dpt_large_384")
print(net_w, net_h, device)
onnx_model_path = 'weights/dpt_large_384.onnx'
filename = 'input/input.jpg'
#image = read_image(filename)
#input_tensor = transform({"image": image})["image"]
input_tensor = prepare_input(filename, net_w, net_h)
print('start prediction')
start = time.time()
session = rt.InferenceSession(onnx_model_path)
model_inputs = session.get_inputs()
input_shape = model_inputs[0].shape

model_outputs = session.get_outputs()
output_shape = model_outputs[0].shape
output_names = [model_outputs[i].name for i in range(len(model_outputs))]
input_names = session.get_inputs()[0].name
output_names = session.get_outputs()[0].name
#input_tensor = np.resize(input_tensor,(1, 3, net_w, net_h))
print(input_names, output_names, input_shape, output_shape, input_tensor.shape)
outputs = session.run(output_names, {input_names: input_tensor})[0]
print(time.time() - start)
depth_map = np.squeeze(outputs, net_w, net_h)
print(outputs.shape, depth_map.shape)
"""
def inference(opt):
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




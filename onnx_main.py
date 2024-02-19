import os
import glob
import torch
import utils
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize
from torchvision import transforms
import argparse
from shutil import copyfile
import fileinput
import sys

from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from onnxs.onnx_convert import convert
from onnxs.onnx_inference import inference

sys.path.append(os.getcwd() + '/..')

def modify_file():
    modify_filename = 'midas/blocks.py'
    copyfile(modify_filename, modify_filename+'.bak')

    with open(modify_filename, 'r') as file :
      filedata = file.read()

    filedata = filedata.replace('align_corners=True', 'align_corners=False')
    filedata = filedata.replace('import torch.nn as nn', 'import torch.nn as nn\nimport torchvision.models as models')
    filedata = filedata.replace('torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")', 'models.resnext101_32x8d()')

    with open(modify_filename, 'w') as file:
      file.write(filedata)
      
def restore_file():
    modify_filename = 'midas/blocks.py'
    copyfile(modify_filename+'.bak', modify_filename)

#modify_file()

#restore_file()

class MidasNet_preprocessing(MidasNet):
    """Network for monocular depth estimation.
    """
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        x.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

        return MidasNet.forward(self, x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='convert', help='convert or inference')
    parser.add_argument('--model_path', type=str, default='weights/model-f6b98070.pt', help='pytorch midas weight model')
    parser.add_argument('-i', '--filename', type=str, default='input/input.jpg', help='input image name')
    parser.add_argument('-o', '--out_name', type=str, default='output/result_onnx', help='optput image name, must png not jpg')
    parser.add_argument('--onnx_model_path', type=str, default='weights/model-f6b98070.onnx', help='onnx midas weight model')
    opt = parser.parse_args()
    if opt.mode=='convert':
        MODEL_PATH = opt.model_path
        convert(MODEL_PATH, MidasNet_preprocessing)
    else:
        inference(opt)
    

import argparse
import cv2
import torch
import utils
from midas.dpt_depth import DPTDepthModel
from midas.midas_net_custom import MidasNet_small
from midas.midas_net import MidasNet
import os
import requests
import gc

def download_file(url, folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Get the file name from the URL
    file_name = url.split("/")[-1]

    # Combine the folder path and file name to get the full file path
    file_path = os.path.join(folder_path, file_name)

    # Check if the file already exists in the folder
    if os.path.exists(file_path):
        print(f"File already downloaded: {file_path}")
    else:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the file and write the content from the response
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded and saved to: {file_path}")
        else:
            print(f"Failed to download the file. HTTP status code: {response.status_code}")

# Custom normalization layer
class NormalizationLayer(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

def patchUnflatten():
    import torch.nn as nn

    class View(nn.Module):
        def __init__(self, dim,  shape):
            super(View, self).__init__()
            self.dim = dim
            self.shape = shape

        def forward(self, input):
            new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
            return input.view(*new_shape)

    nn.Unflatten = View

single_model_params = {"dpt_beit_large_512":
      ["weights/dpt_beit_large_512.pt", "beitl16_512", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"],
     
     "dpt_beit_large_384": ["weights/dpt_beit_large_384.pt", "beitl16_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"],
     
     "dpt_beit_base_384": ["weights/dpt_beit_base_384.pt", "beitb16_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"],
     
     "dpt_swin2_large_384": ["weights/dpt_swin2_large_384.pt", "swin2l24_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"],
     
    "dpt_swin2_base_384": ["weights/dpt_swin2_base_384.pt", "swin2b24_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt"],
    
    "dpt_swin2_tiny_256": ["weights/dpt_swin2_tiny_256.pt", "swin2t16_256", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"],
    
    "dpt_swin_large_384": ["weights/dpt_swin_large_384.pt", "swinl12_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt"],
    
    "dpt_next_vit_large_384": ["weights/dpt_next_vit_large_384.pt", "next_vit_large_6m", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt"],
    
    "dpt_levit_224":["weights/dpt_levit_224.pt", "levit_384", "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"],
    
    "dpt_large_384": ["weights/dpt_large_384.pt", "vitl16_384", "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"],
    
    "midas_v21_384": ["weights/midas_v21_384.pt", "", "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"],
    
    "midas_v21_small_256": ["weights/midas_v21_small_256.pt", "", "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"]
    
    }



model_params = [
    {
        "name": "dpt_beit_large_512",
        "path": "weights/dpt_beit_large_512.pt",
        "backbone": "beitl16_512",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
    },
    {
        "name": "dpt_beit_large_384",
        "path": "weights/dpt_beit_large_384.pt",
        "backbone": "beitl16_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"
    },
    {
        "name": "dpt_beit_base_384",
        "path": "weights/dpt_beit_base_384.pt",
        "backbone": "beitb16_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"
    },
    {
        "name": "dpt_swin2_large_384",
        "path": "weights/dpt_swin2_large_384.pt",
        "backbone": "swin2l24_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"
    },
    {
        "name": "dpt_swin2_base_384",
        "path": "weights/dpt_swin2_base_384.pt",
        "backbone": "swin2b24_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt"
    },
    {
        "name": "dpt_swin2_tiny_256",
        "path": "weights/dpt_swin2_tiny_256.pt",
        "backbone": "swin2t16_256",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"
    },
    {
        "name": "dpt_swin_large_384",
        "path": "weights/dpt_swin_large_384.pt",
        "backbone": "swinl12_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt"
    },
    {
        "name": "dpt_next_vit_large_384",
        "path": "weights/dpt_next_vit_large_384.pt",
        "backbone": "next_vit_large_6m",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt"
    },
    {
        "name": "dpt_levit_224",
        "path": "weights/dpt_levit_224.pt",
        "backbone": "levit_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"
    },
    {
        "name": "dpt_large_384",
        "path": "weights/dpt_large_384.pt",
        "backbone": "vitl16_384",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"
    },
    {
        "name": "midas_v21_384",
        "path": "weights/midas_v21_384.pt",
        "backbone": "",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"
    },
    {
        "name": "midas_v21_small_256",
        "path": "weights/midas_v21_small_256.pt",
        "backbone": "",
        "url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    },
]

def convert_all_models(opt):
    for model_param in reversed(model_params):
        modelName = model_param["name"]
        onnxFile = os.path.join(opt.out_folder, modelName + ".onnx")
        if os.path.exists(onnxFile):
            print(f"ONNX model for {modelName} already exists. Skipping...")
            continue

        download_file(model_param["url"], opt.out_folder)
        model_path = model_param["path"]
        device = torch.device("cpu")

        if modelName != "midas_v21_384" and modelName != "midas_v21_small_256":
            patchUnflatten()

        if modelName == "dpt_levit_224":
            model = DPTDepthModel(
                path=model_path,
                backbone=model_param["backbone"],
                non_negative=True,
                head_features_1=64,
                head_features_2=8,
            )
        elif modelName == "midas_v21_384":
            model = MidasNet(model_path, non_negative=True)
        elif modelName == "midas_v21_small_256":
            model = MidasNet_small(
                model_path,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={'expand': True}
            )
        else:
            model = DPTDepthModel(
                path=model_path,
                backbone=model_param["backbone"],
                non_negative=True,
            )

        # specify input size
        if modelName == "dpt_swin2_tiny_256" or modelName == "midas_v21_small_256":
            net_w, net_h = 256, 256
        elif modelName == "dpt_levit_224":
            net_w, net_h = 224, 224
        else:
            net_w, net_h = 384, 384

        # specify mean and std for normalization
        if modelName == "midas_v21_384" or modelName == "midas_v21_small_256":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
        else:
            mean = torch.tensor([0.5, 0.5, 0.5])
            std = torch.tensor([0.5, 0.5, 0.5])

        # insert normalization layer at the beginning of the model
        norm_layer = NormalizationLayer(mean, std)
        model = torch.nn.Sequential(norm_layer, model)

        model.eval()
        torch.onnx.export(
            model,
            torch.rand(1, 3, net_h, net_w, dtype=torch.float),
            onnxFile,
            export_params=True,
            opset_version=15,
            input_names=["input_image"],
            output_names=["output_depth"],
            do_constant_folding=True
        )

        # free memory
        del model
        gc.collect()
    print('Convert all model files to ONNX.')


def convert_specific_models(opt):
    modelName = opt.model_type
    model_params = single_model_params[modelName]
    model_path = model_params[0]
    model_url = model_params[2]
    backbone = model_params[1]
    
    onnxFile = os.path.join(opt.out_folder, modelName + ".onnx")
    if os.path.exists(onnxFile):
        print(f"ONNX model for {modelName} already exists. Skipping...")

    download_file(model_url, opt.out_folder)
    device = torch.device("cpu")

    if modelName != "midas_v21_384" and modelName != "midas_v21_small_256":
        patchUnflatten()

    if modelName == "dpt_levit_224":
        model = DPTDepthModel(
            path=model_path,
            backbone=backbone,
            non_negative=True,
            head_features_1=64,
            head_features_2=8,
        )
    elif modelName == "midas_v21_384":
        model = MidasNet(model_path, non_negative=True)
    elif modelName == "midas_v21_small_256":
        model = MidasNet_small(
            model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={'expand': True}
        )
    else:
        model = DPTDepthModel(
            path=model_path,
            backbone=backbone,
            non_negative=True,
        )

    # specify input size
    if modelName == "dpt_swin2_tiny_256" or modelName == "midas_v21_small_256":
        net_w, net_h = 256, 256
    elif modelName == "dpt_levit_224":
        net_w, net_h = 224, 224
    else:
        net_w, net_h = 384, 384

    # specify mean and std for normalization
    if modelName == "midas_v21_384" or modelName == "midas_v21_small_256":
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
    else:
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])

    # insert normalization layer at the beginning of the model
    norm_layer = NormalizationLayer(mean, std)
    model = torch.nn.Sequential(norm_layer, model)

    model.eval()
    torch.onnx.export(
        model,
        torch.rand(1, 3, net_h, net_w, dtype=torch.float),
        onnxFile,
        export_params=True,
        opset_version=15,
        input_names=["input_image"],
        output_names=["output_depth"],
        do_constant_folding=True
    )

    # free memory
    del model
    gc.collect()
    print(f'Convert {modelName} model files to ONNX.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='single', help='single or all')
    parser.add_argument('-t', '--model_type',
                        default='dpt_large_384',
                        help='Model type: dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, dpt_levit_224, dpt_large_384, midas_v21_384, midas_v21_small_256')
    parser.add_argument('-o', '--out_folder', type=str, default='weights', help='optput flder')
    opt = parser.parse_args()
    if opt.mode=='single':
        convert_specific_models(opt)
    else:
        convert_all_models(opt)
        
    

 

import argparse
import sys
import time
import cv2
import numpy as np
import torch
    
from utils import normalize_depth, download_mids, download_img
    
def main(opt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas, transform = download_mids(model_type=opt.mode)
    midas.to(device)
    midas.eval()
    
    img = download_img(opt)
    print(img.shape) 
    start_time = time.time()
    input_batch = transform(img).to(device)
    print("input_batch", input_batch.shape)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_img = prediction.cpu().numpy()
    output_norm = cv2.normalize(depth_img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #output_norm = normalize_depth(output_norm, bits=2)
    colored_depth = (output_norm*255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_MAGMA)
    
    print("Prediction took {:.2f} seconds".format(time.time() - start_time))

    cv2.imwrite(opt.out_name+'.jpg', output_norm)
    cv2.imwrite(opt.out_name+'_colored.jpg', colored_depth)
    print('prediction succeeded!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DPT_Large', help='DPT_Large / DPT_Hybrid/ MiDaS_small')
    parser.add_argument('-i', '--filename', type=str, default='input/input.jpg', help='input image name')
    parser.add_argument('-o', '--out_name', type=str, default='output/result', help='optput image name, must png not jpg')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise



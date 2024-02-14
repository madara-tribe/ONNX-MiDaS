import argparse
import cv2
import torch
import urllib.request
    
from utils import normalize_depth, download_mids, download_img
    
def main(opt):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas, transform = download_mids(model_type=opt.mode)
    midas.to(device)
    midas.eval()
    img = download_img(opt.filename)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_img = prediction.cpu().numpy()
    output = normalize_depth(depth_img, bits=2)
    cv2.imwrite(opt.out_name, output)
    print('prediction succeeded!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DPT_Large', help='DPT_Large / DPT_Hybrid/ MiDaS_small')
    parser.add_argument('-i', '--filename', type=str, default='input/input.jpg', help='input image name')
    parser.add_argument('-o', '--out_name', type=str, default='output/result.png', help='optput image name, must png not jpg')
    opt = parser.parse_args()
    try:
        main(opt)
    except KeyboardInterrupt:
        sys.exit(1)
        raise



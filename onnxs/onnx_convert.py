import torch
import numpy as np
import sys
import ntpath

def convert(model_path, MidasNet_preprocessing):
    H = W = 384
    OPSET_VERSION = 11 #9
    """Run MonoDepthNN to compute depth maps.

    Args:
        model_path (str): path to saved model
    """
    print("initialize")
    # load network
    model = MidasNet_preprocessing(model_path, non_negative=True)

    model.eval()

    print("start processing")

    # input
    img_input = np.zeros((3, H, W), np.float32)

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_input.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    torch.onnx.export(model, sample, "weights/"+ntpath.basename(model_path).rsplit('.', 1)[0]+'.onnx', opset_version=OPSET_VERSION)
    print("finished")




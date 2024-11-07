import cv2
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net_custom import MidasNet_small
from midas.midas_net import MidasNet

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

def draw_depth(depth_map, w, h):
    set_min_depth = 0
    set_max_depth = 100
    # Normalize estimated depth to color it
    min_depth = depth_map.min()
    max_depth = depth_map.max()
    min_depth = min_depth if min_depth < set_min_depth else min_depth
    max_depth = max_depth if max_depth > set_max_depth else max_depth

    print(min_depth, max_depth)
    norm_depth_map = 255 * (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map = 255 - norm_depth_map

    # Normalize and color the image
    color_depth = cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1),
                                    cv2.COLORMAP_JET)
    #print(min_depth, max_depth, color_depth.min(), color_depth.max())
    # Resize the depth map to match the input image shape
    return cv2.resize(color_depth, (w, h))


def call_transform(model_type="dpt_large_384"):
    net_w, net_h = 384, 384
    keep_aspect_ratio = False
    if model_type == "midas_v21_384" or model_type == "midas_v21_small_256":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        resize_mode = "upper_bound"
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        resize_mode = "minimal"
    normalization = NormalizeImage(
            mean=mean, std=std
        )
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    return transform, net_w, net_h


class DPTDepthModel_prepro(DPTDepthModel):
    """Network for monocular depth estimation.
    """
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        return DPTDepthModel.forward(self, x)
    
class MidasNet_small_prepro(MidasNet_small):
    """Network for monocular depth estimation.
    """
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        return MidasNet_small.forward(self, x)
        
        
class MidasNet_prepro(MidasNet):
    """Network for monocular depth estimation.
    """
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        return MidasNet.forward(self, x)

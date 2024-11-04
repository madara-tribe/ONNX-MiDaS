# Midas Monocular Depth Estimation

<img src="https://github.com/madara-tribe/MidasDepthEstimater/assets/48679574/a28fd8f1-59b3-4b20-bec4-da3c1e1639d6" width="700px" height="400px"/>


## Version
```
Mac OS CPU
Python 3.8.20
pytorch 2.2.2
tourchvision 0.17.2
onnx 1.7.0 
onnxruntime 1.16.3
```

## Model Type For ONNX
```txt
dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, dpt_large_384, midas_v21_384, midas_v21_small_256'
```
## ONNX Convert
```
# all model convert
$ python3 onnx_convert.py -m all

# specific model convert
$ python3 onnx_convert.py -m single -t <model_type>
```

## ONNX Inferece
```
$ python3 onnx_predict.py -p <model path> -i <input file> -o <output file>
```

## result (dpt_large_384)

<img src="https://github.com/user-attachments/assets/b85f6337-3c9f-4d63-b55f-e690e5c4ad3c" width="400px" height="300px"/>


# References
- [sentis-MiDaS]([https://github.com/isl-org/MiDaS](https://huggingface.co/julienkay/sentis-MiDaS/blob/main/README.md))

- [ONNX-SCDepth-Monocular-Depth-Estimation]([https://github.com/isl-org/MiDaS/tree/master/tf](https://github.com/ibaiGorordo/ONNX-SCDepth-Monocular-Depth-Estimation))

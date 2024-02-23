# Midas Depth Estimater

## Version
```
cuda 12.2
Driver Version: 535.15
Python 3.8.10
pytorch 2.2.0+cu121
tourchvision 0.17.0+cu121
onnx 1.7.0
onnxruntime 1.5.2
```

## how to use
### pytorch 
```
# pytorch inference
$ python3 midas_predict.py --mode <model_version> -i <input path> -o <output path>

# pytorch realtime inference
$ python3 realtime_predict.py -m <model_version> -onnx none -v <video path> -o <output path> -s <plot show>
```

### ONNX
```
# onnx convert
$ python3 onnx_main.py --mode convert --model_path <pytorch weight path>

# onnx inferece
$ python3 onnx_main.py --mode inference --model_path <pytorch weight path> -i <input path> -o <output path> --onnx_model_path <onnx path>

# onnx realtime inferece
$ python3 realtime_predict.py -onnx <onnx_model_path> -v <video path> -o <output path> -s <plot show>
```

## result

### Pytorch (midas-3.1 Large)

<img src="https://github.com/madara-tribe/MidasDepthEstimater/assets/48679574/551b5e10-c0bd-493b-a2b7-fbe43a7228a7" width="500px" height="500px"/>

### ONNX (Midas-2.1)



# References
・[MiDaS-github](https://github.com/isl-org/MiDaS)
・[Midas-onnx](https://github.com/isl-org/MiDaS/tree/master/tf)


import onnx
from onnxsim import simplify
onnx_file = 'weights/dpt_large_384_ver4.onnx'

# predict model type
model_onnx1 = onnx.load(onnx_file)
model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
onnx.save(model_onnx1, 'model1.onnx')

# optimize model structure
model_onnx2 = onnx.load(onnx_file)
model_simp, check = simplify(model_onnx2)
onnx.save(model_simp, 'model2.onnx')

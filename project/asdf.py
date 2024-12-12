import onnx
import numpy as np

model = onnx.load("model.onnx")

for node in model.graph.node:
    print(node.name)

data = np.load("input_tensor.npy")

print(data.shape)
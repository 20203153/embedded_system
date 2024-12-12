import onnx
import numpy as np

model = onnx.load("model.onnx")

for node in model.graph.node:
    print(node.name)

data = np.load("input_tensor.npy")
reshape = np.transpose(data, (0, 3, 2, 1))

print(f"origin: {data.shape} / reshape: {reshape.shape}")
np.save("input_tensor_reshape.npy", reshape)
import torch
import torch.nn as nn
import numpy as np

x_data = np.array([[-1.1, 2.7, 4.3]], dtype=np.float32)
x_data = torch.from_numpy(x_data)

linear = nn.Linear(3, 1)
# torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)

hypothesis_step = torch.relu(torch.sign(linear(x_data)))
hypothesis_sigmoid = torch.sigmoid(torch.sign(linear(x_data)))
hypothesis_ReLU = torch.relu(linear(x_data))

print(f'Step function: {hypothesis_step}')
print(f'Sigmoid function: {hypothesis_sigmoid}')
print(f'ReLU function: {hypothesis_ReLU}')


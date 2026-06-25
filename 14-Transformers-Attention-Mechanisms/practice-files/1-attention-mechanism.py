print("---------------NUMPY-----------------")
import numpy as np

# Define Queris, Keys and Values
queries_np = np.array([[1,0,1],[0,1,1]])
keys_np = np.array([[1,0,1],[1,1,0],[0,1,1]])
values_np = np.array([[10,0],[0,10],[5,5]])

# COmpute attention scores
scores_np = np.dot(queries_np, keys_np.T)

# Apply softmax to normalize scores
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights_np = softmax(scores_np)

# Compute the weighted sum of values
context_np = np.dot(attention_weights_np, values_np)

print("Attention weights: \n",attention_weights_np)
print("Context vector: \n",context_np)



print("---------------PYTORCH-----------------")
import torch
import torch.nn.functional as F

# Define Queris, Keys and Values
queries_py = torch.tensor([[1.0,0.0,1.0],[0.0,1.0,1.0]])
keys_py = torch.tensor([[1.0,0.0,1.0],[1.0,1.0,0.0],[0.0,1.0,1.0]])
values_py = torch.tensor([[10.0,0.0],[0.0,10.0],[5.0,5.0]])

# Compute attention scores
scores_py = torch.matmul(queries_py,keys_py.T)

# Apply softmax to normalize scores
attention_weights_py = F.softmax(scores_py, dim=1)

# Compute weighted sum of values
context_py = torch.matmul(attention_weights_py,values_py)

print("Attention weights: \n",attention_weights_py)
print("Context vector: \n",context_py)

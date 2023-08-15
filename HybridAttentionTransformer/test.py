import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import time

window_size = 16
overlap=.5
expanded_window = int(window_size * (overlap+1))
C = 2*window_size-1
D = 2*expanded_window-1
B = torch.zeros(C, D)
positions = torch.arange(C).unsqueeze(1)
divisor = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
angles = positions * divisor
B[:, 0::2] = torch.sin(angles)
B[:, 1::2] = torch.cos(angles) if B.shape[1] % 2 == 0 else torch.cos(angles[:,:-1])
xrows = torch.arange(1,window_size+1,1/window_size)
xcols = torch.arange(1,expanded_window+1,1/expanded_window)
x = (xcols[None, :]-xrows[:, None]).int()
yrows = torch.concat([torch.arange(1,window_size+1)] * window_size)
ycols = torch.concat([torch.arange(1,expanded_window+1)] * expanded_window)
y = (ycols[None, :]-yrows[:, None])
embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)
print(embeddings.shape)

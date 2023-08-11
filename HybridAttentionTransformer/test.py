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



window_size=32
C = 2*window_size-1
B = torch.zeros(C, C)
positions = torch.arange(C).unsqueeze(1)
divisor = torch.exp(torch.arange(0, C, 2) * (-math.log(10000.0) / C))
angles = positions * divisor
B[:, 0::2] = torch.sin(angles)
B[:, 1::2] = torch.cos(angles[:,:-1])
x = torch.arange(1,window_size+1,1/window_size)
x = (x[None, :]-x[:, None]).int()
y = torch.concat([torch.arange(1,window_size+1)] * window_size)
y = (y[None, :]-y[:, None])
B = B[x[:,:], y[:,:]]




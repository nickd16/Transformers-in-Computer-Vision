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

# SWMSA Input -> (B, H, W, C)

class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.embeddings = RelativeSinusoidalEmbeddings()
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        x = self.proj1(x)
        x = rearrange(x, 'b h w (c K) -> b h w c K', K=3)

        if x.shape[1] == 7:
            self.mask = False
        if self.mask:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4,5)) / math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)

        if self.mask:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).cuda()
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask

        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)

        if self.mask:
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1,2))

        return self.proj2(x)
    
class RelativeSinusoidalEmbeddings(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
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
        self.embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings
    
class RelativeLearnedEmbeddings(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        B = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1))
        x = torch.arange(1,window_size+1,1/window_size)
        x = (x[None, :]-x[:, None]).int()
        y = torch.concat([torch.arange(1,window_size+1)] * window_size)
        y = (y[None, :]-y[:, None])
        self.embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings
    
class ChannelAttentionBlock(nn.Module):
    def __init__(self, C, B):
        super().__init__()
        self.conv1 = nn.Conv2d(C, C//B, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C//B, C, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(C, C//B, kernel_size=1)
        self.conv4 = nn.Conv2d(C//B, C, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        r = self.conv2(self.gelu(self.conv1(x)))
        x = self.pooling(r)
        x = self.conv4(self.relu(self.conv3(x)))
        x = F.sigmoid(x)
        x = r * x
        return x


# HAB Block Input -> (B, C, H, W)

class HybridAttentionBlock(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout=0.1, shift=False):
        super().__init__()
        self.alpha = alpha
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.WMSA = ShiftedWindowMSA(embed_dim, num_heads, window_size, shift)
        self.CAB = ChannelAttentionBlock(embed_dim, squeeze_factor)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[2:]
        norm = self.layer_norm(x)
        norm = (self.alpha * self.CAB(norm)) + self.WMSA(norm.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.dropout(x + norm)
        norm = self.layer_norm(x)
        norm = rearrange(norm, 'B C H W -> B (H W) C')
        norm = self.MLP(norm)
        norm = rearrange(norm, 'B (H W) C -> B C H W', H=height, W=width)
        return self.dropout(x + norm)
  
class OverlappingCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ResidualAttentionGroup(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def main():
    x = torch.zeros((64, 200, 224, 224))
    CAB = ChannelAttentionBlock(C=200, B=2)
    x = CAB(x)
    print(x.shape)

if __name__ == '__main__':
    main()
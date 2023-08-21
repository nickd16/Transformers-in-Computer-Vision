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
import os

# SWMSA Input -> (B, C, H, W)

class RelativeSinusoidalEmbeddings(nn.Module):
    def __init__(self, window_size, overlap=0):
        super().__init__()
        expanded_window = int(window_size * (overlap+1))
        C = 2*window_size-1
        D = 2*expanded_window-1
        B = torch.zeros(C, D)
        positions = torch.arange(C).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))
        angles = positions * divisor
        B[:, 0::2] = torch.sin(angles)
        B[:, 1::2] = torch.cos(angles) if (B.shape[0] % 2 == 0 and B.shape[1] % 2 == 0) else torch.cos(angles[:,:-1])
        xrows = torch.arange(1,window_size+1,1/window_size)
        xcols = torch.arange(1,expanded_window+1,1/expanded_window)
        x = (xcols[None, :]-xrows[:, None]).int()
        yrows = torch.concat([torch.arange(1,window_size+1)] * window_size)
        ycols = torch.concat([torch.arange(1,expanded_window+1)] * expanded_window)
        y = (ycols[None, :]-yrows[:, None])
        self.embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.embeddings = RelativeSinusoidalEmbeddings(window_size)

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height, width = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        if x.shape[1] == self.window_size:
            self.mask = False
        if self.mask:
            x = torch.roll(x, (-self.window_size//2, -self.window_size//2), dims=(1,2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4,5)) 
        att_scores /= math.sqrt(h_dim)
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

        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.proj2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
        return x
    
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
        self.relu = nn.ReLU(inplace=True)

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
        norm = self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        norm = (self.alpha * self.CAB(norm)) + self.WMSA(norm)
        x = self.dropout(x + norm)
        norm = self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        norm = rearrange(norm, 'B C H W -> B (H W) C')
        norm = self.MLP(norm)
        norm = rearrange(norm, 'B (H W) C -> B C H W', H=height, W=width)
        return self.dropout(x + norm)

# HAB Block Input -> (B, C, H, W)

class OverlappingCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, overlap_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.lambd = overlap_size
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.embeddings = RelativeSinusoidalEmbeddings(window_size, overlap_size)
        padding = int((overlap_size*window_size)/2)
        self.Mo = int((1 + overlap_size) * window_size)
        self.unfold = nn.Unfold(kernel_size=self.Mo, stride=window_size, padding=padding)

    def forward(self, x):
        h_dim = self.embed_dim / self.num_heads
        height, width = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        Q, K, V = x.chunk(3, dim=4)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        Q = rearrange(Q, 'b (h m1) (w m2) (H E) -> b H h w (m1 m2) E', H=self.num_heads, m1=self.window_size, m2=self.window_size)
        K, V = self.unfold(K.permute(0,3,1,2)), self.unfold(V.permute(0,3,1,2))
        num_windows = int(math.sqrt(K.shape[2]))
        K = rearrange(K, 'B (H c s1 s2) (h w) -> B H h w (s1 s2) c', H=self.num_heads, s1=self.Mo, s2=self.Mo, h=num_windows, w=num_windows)
        V = rearrange(V, 'B (H c s1 s2) (h w) -> B H h w (s1 s2) c', H=self.num_heads, s1=self.Mo, s2=self.Mo, h=num_windows, w=num_windows)
        
        att_scores = (Q @ K.transpose(4,5)) 
        att_scores /= math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)
        att = F.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=self.window_size, m2=self.window_size)
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.proj2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)

        return x
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads, dropout=0.1, overlap_size=0.5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.OCA = OverlappingCrossAttention(embed_dim, num_heads, window_size, overlap_size)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[2:]
        norm = self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        norm = self.OCA(norm)
        x = self.dropout(x + norm)
        norm = self.layer_norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        norm = rearrange(norm, 'B C H W -> B (H W) C')
        norm = self.MLP(norm)
        norm = rearrange(norm, 'B (H W) C -> B C H W', H=height, W=width)
        return self.dropout(x + norm)

class ResidualHybridAttentionGroup(nn.Module):
    def __init__(self, embed_dim, window_size=16, num_heads=6, dropout=0.1, squeeze_factor=3, overlap_size=0.5, alpha=0.01):
        super().__init__()
        self.HAB1 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout)
        self.HAB2 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout, shift=True)
        self.HAB3 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout)
        self.HAB4 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout, shift=True)
        self.HAB5 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout)
        self.HAB6 = HybridAttentionBlock(embed_dim, window_size, num_heads, squeeze_factor, alpha, dropout, shift=True)
        self.OCA = OverlappingCrossAttention(embed_dim, num_heads, window_size, overlap_size)

    def forward(self, x):
        for i in range(1,7):
            layer = getattr(self, f'HAB{i}')
            x = layer(x)
        x = self.OCA(x)
        return x

class HAT_model(nn.Module):
    def __init__(self, ratio=2, C=64, embed_dim=180):
        super().__init__()
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(embed_dim, C*(ratio**ratio), kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(C, C*(ratio**ratio), kernel_size=3, padding=1)
        self.output = nn.Conv2d(C, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.RHAG = ResidualHybridAttentionGroup(embed_dim)
        self.pixelshuffle1 = nn.PixelShuffle(ratio)
        self.pixelshuffle2 = nn.PixelShuffle(ratio)

    def forward(self, x):
        y = self.conv1(x)
        x = self.RHAG(y)
        x = self.conv2(x) + y
        x = self.relu(self.conv3(x))
        x = self.pixelshuffle1(x)
        x = self.conv4(x)
        x = self.pixelshuffle2(x)
        x = self.output(x)
        return x

def main():
    x = torch.zeros((1, 3, 224, 224)).cuda()
    model = HAT_model().cuda()
    model(x)

if __name__ == '__main__':
    main()
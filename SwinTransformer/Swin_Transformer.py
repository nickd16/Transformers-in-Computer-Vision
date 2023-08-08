import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from data_utils import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import time

class SwinEmbedding(nn.Module):
    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.patch_size = patch_size
        self.linear_embedding = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear_embedding(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.relu(self.layer_norm(x))
        return x
    
class PatchMerging(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.linear = nn.Linear(4*C, 2*C)
        self.layer_norm = nn.LayerNorm(2*C)

    def forward(self, x):
        x = rearrange(x, 'b (h s1) (w s2) c -> b h w (s2 s1 c)', s1=2, s2=2)
        return self.layer_norm(self.linear(x))

class RelativeEmbeddings(nn.Module):
    def __init__(self, window_size=7, shift=False):
        super().__init__()
        B = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1))
        x = torch.arange(1,window_size+1,1/window_size)
        x = x.int()
        x = (x[None, :]-x[:, None])
        y = torch.concat([torch.arange(1,window_size+1)] * window_size)
        y = (y[None, :]-y[:, None])
        if shift:
            x = torch.roll(x, (-window_size//2, -window_size//2), (0,1))
            y = torch.roll(y, (-window_size//2, -window_size//2), (0,1))
        self.embeddings = nn.Parameter((B[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings

class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.embeddings = RelativeEmbeddings() if mask==False else RelativeEmbeddings(shift=True)
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
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1=7, m2=7)

        if self.mask:
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1,2))

        return self.proj2(x)

class SwinEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.WMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads)
        self.MLP1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.SWMSA = ShiftedWindowMSA(embed_dim=embed_dim, num_heads=num_heads, mask=True)
        self.MLP2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[1:3]
        res1 = self.dropout(self.WMSA(self.layer_norm(x)) + x)
        x = self.layer_norm(res1)
        x = rearrange(x, 'B h w c -> B (h w) c')
        x = self.MLP1(x)
        x = rearrange(x, 'B (h w) c -> B h w c', h=height, w=width)
        x = self.dropout(x + res1)

        res2 = self.dropout(self.SWMSA(self.layer_norm(x)) + x)
        x = self.layer_norm(res2)
        x = rearrange(x, 'B h w c -> B (h w) c')
        x = self.MLP2(x)
        x = rearrange(x, 'B (h w) c -> B h w c', h=height, w=width)
        x = self.dropout(x + res2)
        return x

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.PatchMerge1 = PatchMerging(96)
        self.PatchMerge2 = PatchMerging(192)
        self.PatchMerge3 = PatchMerging(384)
        self.EncoderBlock1 = SwinEncoderBlock(96, 3)
        self.EncoderBlock2 = SwinEncoderBlock(192, 6)
        self.EncoderBlock3_1 = SwinEncoderBlock(384, 12)
        self.EncoderBlock3_2 = SwinEncoderBlock(384, 12)
        self.EncoderBlock3_3 = SwinEncoderBlock(384, 12)
        self.EncoderBlock4 = SwinEncoderBlock(768, 24)
        self.MLP = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PatchMerge1(self.EncoderBlock1(x))
        x = self.PatchMerge2(self.EncoderBlock2(x))
        x = self.EncoderBlock3_1(x)
        x = self.EncoderBlock3_2(x)
        x = self.EncoderBlock3_3(x)
        x = self.PatchMerge3(x)
        x = self.EncoderBlock4(x)
        x = rearrange(x, 'B h w c -> B (h w) c')
        x = torch.mean(x, dim=1)
        return self.MLP(x)

def main():
    device = torch.device('cuda')
    x = torch.randn((32, 3, 224, 224)).to(device)
    model = SwinTransformer().to(device)
    x = model(x)
    print(x.shape)

    




if __name__ == "__main__":
    main()
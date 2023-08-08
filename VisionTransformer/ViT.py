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

# 5x convolution stem instead of 16x16, s=16 conv linear projection

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, image_size, patch_size, seq_length):
        super().__init__()
        self.Conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1)
        self.Conv2 = nn.Conv2d(48, 96, kernel_size=3, stride=2,  padding=1)
        self.Conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=2,  padding=1)
        self.Conv4 = nn.Conv2d(192, 384, kernel_size=3, stride=2,  padding=1)
        self.Conv5 = nn.Conv2d(384, 768, kernel_size=1)
        self.Batchnorm1 = nn.BatchNorm2d(48)
        self.Batchnorm2 = nn.BatchNorm2d(96)
        self.Batchnorm3 = nn.BatchNorm2d(192)
        self.Batchnorm4 = nn.BatchNorm2d(384)
        self.Relu = nn.ReLU()
        self.class_token = nn.Parameter(torch.randn((1,1,d_model)))
        self.embeddings = nn.Parameter(torch.randn((seq_length,d_model)))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.Relu(self.Batchnorm1(self.Conv1(x)))
        x = self.Relu(self.Batchnorm2(self.Conv2(x)))
        x = self.Relu(self.Batchnorm3(self.Conv3(x)))
        x = self.Relu(self.Batchnorm4(self.Conv4(x)))
        x = self.Conv5(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        token = self.class_token.expand(batch_size, 1, self.class_token.shape[2])
        x = torch.concat([token, x], dim=1)
        x = x + self.embeddings
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj1 = nn.Linear(embed_dim, embed_dim*3)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_dim, seq_dim = x.shape[:2]
        h_dim = int(x.shape[2] / self.num_heads)
        x = self.proj1(x)
        x = x.reshape(batch_dim, seq_dim, self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        K, Q, V = x.chunk(3, dim=3)
        att = F.softmax( (torch.matmul(Q, V.transpose(-1, -2)) / math.sqrt(h_dim)), dim=-1 ) @ V
        x = att.permute(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)
        return self.proj2(x)

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.Dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.MSA = MultiHeadSelfAttention()
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = self.Dropout(self.MSA(self.LayerNorm(x)) + x)
        x = self.Dropout(self.MLP(self.LayerNorm(x)) + x)
        return x

class ViT(nn.Module):
    def __init__(self, d_model=768, image_size = 224, patch_size = 16, nhead=12, hidden_dim=1024, encoder_layers=1, classes=256):
        super().__init__()
        self.seq_length = int((image_size/patch_size)**2+1)
        self.Embedding = EmbeddingLayer(d_model, image_size, patch_size, self.seq_length)
        self.Encoder1 = ViTEncoder(d_model, nhead, hidden_dim)
        self.Encoder2 = ViTEncoder(d_model, nhead, hidden_dim)
        self.Encoder3 = ViTEncoder(d_model, nhead, hidden_dim)
        self.Encoder4 = ViTEncoder(d_model, nhead, hidden_dim)
        self.Hidden1 = nn.Linear(d_model, hidden_dim)
        self.Gelu = nn.GELU()
        self.Output = nn.Linear(hidden_dim, classes)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.Encoder1(x)
        x = self.Encoder2(x)
        x = self.Encoder3(x)
        x = self.Encoder4(x)
        x = self.Gelu(self.Hidden1(torch.mean(x, dim=1)))
        return self.Output(x)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224), antialias=True),
        ToRGB(),
        transforms.Normalize(mean=[0.552, 0.5336, 0.505], std=[0.3157, 0.3123, 0.326])
    ]) 

    dataset = datasets.Caltech256(root='data/', download=False, transform=transform) 
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=True)
    device = torch.device('cuda')

    for i, (x,y) in enumerate(data_loader):
        if i == 1:
            break
        x = x.to(device)
        print(x.shape)
        model = ViT().to(device)
        x = model(x)
        print(x.shape)
        if i == 1:
            break

if __name__ == '__main__':
    main()




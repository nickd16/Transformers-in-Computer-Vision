import torch 
import torch.nn as nn
from einops import rearrange
from data_utils import *
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision
from VisionTransformer.ViT import ViT
from SwinTransformer.Swin_Transformer import SwinTransformer

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224), antialias=True),
        ToRGB(),
        transforms.Normalize(mean=[0.552, 0.5336, 0.505], std=[0.3157, 0.3123, 0.326])
    ]) 

    batch_size = 24
    dataset = datasets.Caltech256(root='data/', download=False, transform=transform) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    device = torch.device('cuda')
    lr = 1e-4
    model = SwinTransformer().to(device)
    #model.load_state_dict(torch.load('weights/conv_stem.pth'))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 5
    best_acc = 0
    best_loss = float('inf')
    for i in range(epochs):
        total_iters = int(dataset.__len__() / batch_size)
        progress_bar = tqdm.tqdm(total=total_iters, desc='Epoch', unit='iter')
        best_acc = total_loss = total_batches = total_correct = 0
        for bidx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y = torch.clamp(y, max=255)
            outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss

            total_batches += batch_size
            _, indexes = torch.max(outputs, dim=1)
            total_correct += (indexes==y).sum()
            acc = total_correct/total_batches

            progress_bar.update(1) 

            if bidx % 200 == 0:
                print(f'Epoch {epochs+1} | Loss {loss:.4f} | Accuracy {acc:.4f}')
    
        progress_bar.close()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'weights/swin_test3.pth')
            print("Saving Current Model Weights")

if __name__ == '__main__':
    main()
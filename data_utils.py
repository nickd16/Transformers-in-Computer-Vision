import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def unnormalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.552, 0.5336, 0.505]).view(3,1,1)
        std = torch.tensor([0.3157, 0.3123, 0.326]).view(3,1,1)
        x = ((x*std)+mean) * 255
        x = x.permute(1,2,0).to(torch.int16).numpy()
        return x

def display(x, Tensor=False):
    if Tensor:
        x = unnormalize(x)
    plt.imshow(x)
    plt.axis('off')
    plt.show()

def display_patches(x: torch.Tensor):
    fig, axes = plt.subplots(nrows=14, ncols=14, figsize=(9,9))
    for i, ax in enumerate(axes.flat):
        patch = rearrange(x[i], '(s1 s2 c) -> c s1 s2', s1=16, s2=16)
        patch = unnormalize(patch)
        ax.imshow(patch)
        ax.axis('off')
    fig.tight_layout()
    plt.show()

class rescale():
    def __init__(self, const):
        self.const = const
    
    def __call__(self, x):
        return x / self.const
    
class ToRGB():
    def __call__(self, x: torch.tensor) -> torch.tensor:
        if x.shape[0] == 1:
            return x.expand(3, -1, -1)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    ToRGB(),
    transforms.Normalize(mean=[0.5520, 0.5336, 0.5050], std=[0.3157, 0.3123, 0.3260])
]) 

def mean_std():
    dataset = datasets.Caltech256(root='data/', download=False, transform=transform)  
    length = dataset.__len__()
    device = torch.device('cuda')

    total = torch.zeros((1, 3)).cuda()

    train_loader = DataLoader(dataset, batch_size=1)
    for i, (x,y) in enumerate(train_loader):
        x = x.to(device)
        total += torch.sum(x, dim=(2,3))
    mean = total / (224*224*length)
    print(mean)

    total = torch.zeros((1,3)).cuda()

    for i, (x,y) in enumerate(train_loader):
        x = x.to(device)
        diff = (x - mean.view(1,3,1,1))
        total += torch.sum(diff**2, dim=(2,3))

    std = torch.sqrt(total / (224*224*length))
    print(std)

def main():
    dataset = datasets.Caltech256(root='data/', download=False, transform=transform) 
    x, y = dataset[0]
    print(x.shape)
    # display(x, True)
    # print(x.shape)
    # x = rearrange(x, 'c (h s1) (w s2) -> (h w) (s1 s2 c)', s1=16, s2=16)
    # display_patches(x)

if __name__ == '__main__':
    main()
 
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import random

class randomBlur(object):
    def __init__(self, prob=90):
        self.prob = prob

    def __call__(self, x):
        r = random.randint(1,100)
        if r < self.prob:
            r = random.randint(1,100)
            x = cv.GaussianBlur(x, (5,5), 0) if r < 75 else cv.blur(x, (5,5))
        return x

class DIV2k(Dataset):
    def __init__(self, train=True):
        path = 'DIV2K_'
        path += 'train_HR' if train else 'valid_HR'
        self.X = []
        self.Y = []
        for file in os.listdir(path):
            img = cv.imread(path+'/'+file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            self.Y.append(cv.resize(img, (1792, 1792)))
            self.X.append(cv.resize(img, (224,224)))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.blur = randomBlur()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = self.blur(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        x = x.permute(2, 0, 1)
        y = y.permute(2, 0, 1)
        x = x / 255.0
        x = self.normalize(x)
        return (x,y)
    
def display(image, image_real, normalize=True):
    image_real = image_real.squeeze(0)
    image = image.squeeze(0)
    image = image.permute(1,2,0)
    image_real = image_real.permute(1,2,0)
    image = image.int().numpy()
    image_real = image_real.int().numpy()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[1].imshow(image_real)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    dataset = DIV2k(train=False)
    for i in range(50):
        x,y = dataset[0]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        display(x,y)

if __name__ == '__main__':
    main()
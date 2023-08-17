import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2 as cv
import os

class DIV2k(Dataset):
    def __init__(self, train=True):
        path = 'DIV2K_'
        path += 'train_HR' if train else 'valid_HR'
        self.X = []
        self.Y = []
        for file in os.listdir(path):
            img = cv.imread(path+'/'+file)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            self.Y.append(cv.resize(img, (896, 896)))
            self.X.append(cv.GaussianBlur((cv.resize(img, (224,224))), (5,5), 0))
        self.normalize = transforms.Normalize(mean=[], std=[])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        # x = x / 255.0
        # x = self.normalize(x)
        return (x,y)
    
def main():
    dataset = DIV2k(train=True)
    for i in range (100):
        image, image_real = dataset[i]
        image = image.int().numpy()
        image_real = image_real.int().numpy()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].axis('off')
        axes[1].imshow(image_real)
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()

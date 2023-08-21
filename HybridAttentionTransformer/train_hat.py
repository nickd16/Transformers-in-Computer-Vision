import torch 
import torch.nn as nn
from einops import rearrange
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from HAT import HAT_model
from DIV2K import DIV2k
import random
import cv2 as cv
import torchvision.transforms as transforms

def train():
    batch_size = 2
    dataset = DIV2k(train=True) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    device = torch.device('cuda')
    lr = 2e-5
    model = HAT_model().to(device)
    model.load_state_dict(torch.load('weights/1792x1792_2.pth'))
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 1
    best_acc = 0
    best_loss = float('inf')
    for i in range(epochs):
        total_iters = int(dataset.__len__() / batch_size)
        progress_bar = tqdm.tqdm(total=total_iters, desc='Epoch', unit='iter')
        best_acc = total_loss = total = correct = 0
        for bidx, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss

            total += y.numel()
            correct += (outputs.int() == y).sum()
            acc = correct/total

            progress_bar.update(1) 

            if (bidx+1) % 400 == 0:
                print(f'Epoch {epochs+1} | Loss {total_loss/(bidx+1):.4f} | Accuracy {acc:.4f}')
    
        progress_bar.close()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'weights/1792x1792_2.pth')
            print("Saving Current Model Weights")

def test():
    batch_size = 2
    dataset = DIV2k(train=False) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    device = torch.device('cuda')
    model = HAT_model().to(device).eval()
    model.load_state_dict(torch.load('weights/1792x1792_2.pth'))
    criterion = nn.L1Loss()

    epochs = 1
    best_acc = 0
    best_loss = float('inf')
    with torch.no_grad():
        for i in range(epochs):
            total_iters = int(dataset.__len__() / batch_size)
            progress_bar = tqdm.tqdm(total=total_iters, desc='Epoch', unit='iter')
            best_acc = total_loss = total = correct = 0
            for bidx, (x,y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)

                loss = criterion(outputs, y)
                total_loss += loss
                total += y.numel()
                correct += (outputs.int() == y).sum()
                acc = correct/total

                progress_bar.update(1) 

                if (bidx+1) % 25 == 0:
                    print(f'Epoch {epochs+1} | Loss {total_loss/(bidx+1):.4f} | Accuracy {acc:.4f}')
            progress_bar.close()

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
    model = HAT_model().cuda()
    model.load_state_dict(torch.load('weights/1792x1792_2.pth'))
    for i in range(16):
        r = random.randint(1,99)
        x,y = dataset[r]
        x, y = x.cuda().unsqueeze(0), y.unsqueeze(0)
        output = model(x)
        output = output.cpu()
        x = x.cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        x = ((x*std) + mean) * 255
        display(output,x,False)

def image_test():
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = HAT_model().cuda()
    model.load_state_dict(torch.load('weights/1792x1792_2.pth'))
    x = cv.imread('test4.png')
    y = cv.resize(x, (1792, 1792))
    y = cv.cvtColor(y, cv.COLOR_BGR2RGB)
    y = torch.tensor(y).unsqueeze(0).float()
    y = y.permute(0, 3, 1, 2)
    x = cv.cvtColor(x, cv.COLOR_BGR2RGB)
    x = cv.resize(x, (224,224))
    x = torch.tensor(x).cuda().unsqueeze(0).float()
    x = x.permute(0, 3, 1, 2)
    x = x / 255.0
    x = normalize(x)
    output = model(x)
    x = x.cpu()
    x = ((x*std) + mean) * 255
    output = output.cpu()
    display(output, y, False)

if __name__ == '__main__':
    main()
#Chidilim(Dilly) Ejeh - 20158835
#Maahum Khan - 20232476
import sys

import datetime
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import autoencoderMLPL4Layer

transformm = transforms.Compose([
    transforms.ToTensor()])

testing_data = MNIST('./data/mnist',
                     train=False,
                     transform=transformm,
                     download=True
                     )
training_data = MNIST('./data/mnist',
                     train=True,
                     transform=transformm,
                     download=False
                     )


idx = int(input("Enter a number between 0 and 5999: "))

device = torch.device("cpu")

test_loader = torch.utils.data.DataLoader(testing_data, batch_size=2048, shuffle=False)




def test(noise, mymodel, loss_fn, test_loader, device):

    #train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=transformm)
    #test_set = datasets.MNIST('./data/mnist', train=False, download=True, transform=transformm)
    mymodel.eval()
    loss_fn = loss_fn
    losses = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            if noise == True:
                noisedImgs = imgs + torch.rand(imgs.shape)
            #imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs.to(device=device, dtype=torch.float32)
            imgs = imgs.view(imgs.size(0), -1)
            y = mymodel(imgs)
            y = y.view(y.size(0), -1)
            loss = loss_fn(imgs, y)
            losses += [loss.item()]


       # original_img = imgs[idx].cpu().numpy().reshape(28,28)  # Convert tensor to NumPy array
        #reconstructed_img = y[idx].cpu().numpy().reshape(28,28)  # Convert tensor to NumPy array
            torch.save(mymodel.state_dict(), args.l)


            if noise == True:
                f = plt.figure()  # Adjust the figure size as needed
                f.add_subplot(1, 3, 1)
                plt.imshow(imgs[idx].cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                plt.title("Original Image")

                f.add_subplot(1, 3, 2)
                plt.imshow(noisedImgs[idx].cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                plt.title("Noise Image")

                f.add_subplot(1, 3, 3)
                plt.imshow(y[idx].cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                plt.title("Reconstructed Image")
                plt.show()
            else:
                f = plt.figure()  # Adjust the figure size as needed
                f.add_subplot(1, 2, 1)
                plt.imshow(imgs[idx].cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
                plt.title("Original Image")
                f.add_subplot(1, 2, 2)
                plt.imshow(y[idx].cpu().numpy().reshape(28,28), cmap='gray', vmin=0, vmax=1)
                plt.title("Reconstructed Image")
                plt.show()

    # Plot the original and reconstructed images side by side

def BNInter(mymodel,device):

    mymodel.eval()
    with torch.no_grad():

        Btensor = training_data[0],training_data[1]

        train_loader = torch.utils.data.DataLoader(Btensor, shuffle=False)

        Btensor1 = train_loader.dataset[0][0]
        Btensor2 = train_loader.dataset[1][0]

        Btensor1 = Btensor1.view(Btensor1.size(0),-1)
        Btensor2 = Btensor2.view(Btensor2.size(0), -1)

        Btensor1_ = Btensor1.to(device =device, dtype = torch.float32)
        Btensor2 = Btensor2.to(device=device, dtype=torch.float32)



        encode1 = mymodel.encode(Btensor1)
        encode2 = mymodel.encode(Btensor2)

        n = int(input("Enter a desired number of interpolation steps. The number should be between 2 and 10: "))
        f = plt.figure()

        for i in range(n):
            alpha = i/(n-1)
            interpolatedVersion = torch.lerp(encode1, encode2, alpha)
            result = mymodel.decode(interpolatedVersion)

         # Adjust the figure size as needed
            f.add_subplot(1, n, i+1)
            plt.imshow(result.cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        #plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=str, default="MLP.8.pth", help="MLP Path")
    args = parser.parse_args()

    loss_fnn = nn.MSELoss()
    losses = []

    #mymodell = autoencoderMLPL4Layer(N_input=784, N_bottleneck=8, N_output=784).to(device)
    mymodell = autoencoderMLPL4Layer()
    mymodell.load_state_dict(torch.load(args.l))
    test(False,mymodel=mymodell, loss_fn=loss_fnn, test_loader=test_loader, device=device)

    mymodel2 = autoencoderMLPL4Layer()
    mymodel2.load_state_dict(torch.load(args.l))
    test(True, mymodel=mymodel2, loss_fn=loss_fnn, test_loader=test_loader,device=device)

    mymodel3 = autoencoderMLPL4Layer()
    mymodel3.load_state_dict(torch.load(args.l))
    BNInter(mymodel3,device=device)


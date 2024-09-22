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
import torch.optim as optim
from model import autoencoderMLPL4Layer


def train(n_epochs, optimizer, model, loss_fn, train_dataset, val_dataset, scheduler, device, lossplot):

    #MAYBE TAKE THIS OUT ------- torch.save(model.state_dict(), args.s)

    print('training....')
    #model.train()
    losses_train = []
    losses_val = []

    for epoch in range(1,n_epochs+1):
        train_loader = iter(DataLoader(train_dataset, batch_size=2048,shuffle=True,drop_last=True))
        model.train()
        loss_train = 0.0
        for imgs, _ in train_loader:
            # Reshape the images to be 1D (784 elements)
            imgs = imgs.view(imgs.size(0), -1)
            # Move each image tensor to the specified device
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            Tloss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            Tloss.backward()
            optimizer.step()
            loss_train += Tloss.item()
        scheduler.step(loss_train)
        losses_train += [loss_train/len(train_loader)]
        print('{} Epoch {}, TRAINING loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))

    plt.figure(2, figsize=(12,7))
    plt.plot(losses_train, label='Training')
    plt.plot(losses_val, label='Validating')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.title('Training and Validation Loss Over Time')
    plt.savefig(lossplot)
    plt.show()

    return losses_train
#end of def train(...)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", type=int, default=8, help="Bottleneck")
    parser.add_argument("-e", type=int, default=50, help="Epochs")
    parser.add_argument("-b", type=int, default=1024, help="Batch size")
    parser.add_argument("-s", type=str, default="MLP.8.pth", help="MLP Path")
    parser.add_argument("-p", type=str, default="loss.MLP.8.png", help="MLP png")
    args = parser.parse_args()

    device = torch.device("cpu")

    transformm = transforms.Compose([transforms.ToTensor()])

    mymodel = autoencoderMLPL4Layer(N_input=784,N_bottleneck=args.z,N_output=784).to(device)
    # Maybe try replacing (2048, 1, 1, 784) with (model, (1, 784))
    summary(mymodel, (2048, 1, 1, 784))

    learning_rate = 1e-4
    loss_fn = nn.MSELoss()

    theoptimizer = optim.Adam(mymodel.parameters(), lr=learning_rate)
    thescheduler = torch.optim.lr_scheduler.StepLR(theoptimizer,step_size=30,gamma=0.1, last_epoch=-1)

    training_data = MNIST('./data/mnist',
        train=True,
        transform=transformm,
        download=True
    )
    #train_loader = torch.utils.data.DataLoader(training_data, batch_size=1024, shuffle=True)
    #train(2, theoptimizer, mymodel, loss_fn, train_loader, thescheduler, device=device)

    validation_data = MNIST('./data/mnist',
                          train=False,
                          transform=transformm,
                          download=True
    )
    #validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1024, shuffle=False)
    # PUT args.e AS FIRST PARAMETER WHEN ARGS THING ADDED
    #losses_train = (

    losses_train = train(args.e, theoptimizer, mymodel, loss_fn, training_data, validation_data,thescheduler, device=device, lossplot=args.p)

    torch.save(mymodel.state_dict(), args.s)

   # plt.figure()
    #plt.plot(losses_train, label='training')
   # plt.xlabel('Epoch')
   # plt.ylabel('Training Loss')
    #plt.title('Training Loss Over Time')
   # plt.show()

    # TO RUN, TYPE THE FOLLOWING IN PYCHARM TERMINAL: python train.py --z 8 --e 50 --b 2048 --s MLP.8.pth --p loss.MLP.8.png
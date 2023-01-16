import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transformations
transform = T.Compose([ T.ToTensor(),
                        T.Normalize(.5,.5),
                       ])

# import data and transform
dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)

# transform to dataloader
batchsize   = 100
data_loader = DataLoader(dataset,batch_size=batchsize,drop_last=True)

class discriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64,128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128,256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256,512, 3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(512,  1, 3, 1, 0, bias=False)

        # batchnorm
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        return torch.sigmoid(self.conv5(x)).view(-1,1)

class generatorNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self.conv1 = nn.ConvTranspose2d(100,512, 4, 1, 0,bias=False)
        self.conv2 = nn.ConvTranspose2d(512,256, 3, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d(256,128, 3, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d(128, 28, 3, 2, 0, bias=False)
        self.conv5 = nn.ConvTranspose2d(28,   1, 2, 1, 0, bias=False)

        # batchnorm
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d( 28)


    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.tanh( self.conv5(x) )
        return x

lossfun = nn.BCELoss()

dnet = discriminatorNet().to(device)
gnet = generatorNet().to(device)

d_optimizer = torch.optim.Adam(dnet.parameters(), lr=.0005)
g_optimizer = torch.optim.Adam(gnet.parameters(), lr=.0005)

num_epochs = 15

losses  = []

for epoch in range(num_epochs):

    for data,_ in data_loader:

        # data to GPU
        data = data.to(device)

        # create labels for real and fake images
        real_labels = torch.ones(batchsize,1).to(device)
        fake_labels = torch.zeros(batchsize,1).to(device)



        # train discriminator

        # forward pass and loss for real pictures
        pred_real   = dnet(data)                     
        d_loss_real = lossfun(pred_real,real_labels)
        
        # forward pass and loss for fake pictures
        fake_data   = torch.randn(batchsize,100,1,1).to(device)
        fake_images = gnet(fake_data)
        pred_fake   = dnet(fake_images)
        d_loss_fake = lossfun(pred_fake,fake_labels)

        # combine losses
        combined_loss = d_loss_real + d_loss_fake

        # backprop
        d_optimizer.zero_grad()
        combined_loss.backward()
        d_optimizer.step()



        # train generator

        # create fake images and compute loss
        fake_images = gnet(torch.randn(batchsize,100,1,1).to(device))
        pred_fake   = dnet(fake_images)

        # compute loss
        g_loss = lossfun(pred_fake,real_labels)

        # backprop
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()


        # append losses
        losses.append([combined_loss.item(),g_loss.item()])


    # print current epoch
    print(f'Finished epoch {epoch+1}/{num_epochs}')


# convert loss to numpy array
losses  = np.array(losses)
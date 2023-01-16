import numpy as np

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import matplotlib.pyplot as plt

st.title("Making MNIST data with a CNN GAN")
cols = st.columns(2)
cols[0].markdown("This project is focused on utilizing a Generative Adversarial Network (GAN) in combination with a Convolutional Neural Network (CNN) to generate new images of handwritten digits that resemble those found in the MNIST dataset. The GAN is made up of two neural networks: a generator and a discriminator. The generator's job is to create new images, whereas the discriminator's role is to evaluate the generated images and determine whether they are real or fake. Essentially, the generator and discriminator are in a constant battle, with the generator creating new images to fool the discriminator, and the discriminator trying to identify the fake images. The ultimate goal is to train the GAN so that it can produce high-quality images that are indistinguishable from real MNIST images.")
cols[1].image("models/MNIST CNN GAN/MNIST CNN GAN loss.png")

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

gnet = generatorNet()
gnet.load_state_dict(torch.load("models/MNIST CNN GAN/MNISTGenerator.pt",map_location=torch.device('cpu')))


gnet.eval()
placeholder = st.empty()
while True:
    fig,ax = plt.subplots(1,2,figsize=(12,6))
    fake_data = torch.randn(1,100,1,1)
    fake_MNIST = gnet(fake_data)

    # visualize
    ax[0].imshow(fake_data.squeeze().detach().numpy().reshape(10,10),cmap='gray')
    ax[0].axis('off')
    ax[0].set_title("Before")

    ax[1].imshow(fake_MNIST[0,:,].detach().squeeze(),cmap='gray')
    ax[1].axis('off')
    ax[1].set_title("After")

    placeholder.pyplot(fig)
    time.sleep(1.5)


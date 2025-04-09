# Autoencoder Implementation with PyTorch

This repository contains an implementation of an autoencoder using PyTorch, trained on the MNIST dataset of handwritten digits. The objective is to learn a compressed latent representation of images and reconstruct them with minimal loss.

## Overview
An autoencoder is an unsupervised neural network that learns to compress input data into a latent space and then reconstruct it back to the original form. This project demonstrates the construction and training of an autoencoder to work with the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.

![image](https://github.com/user-attachments/assets/1e0fef71-4e14-42b6-b544-a73f05406e1f)

## Model Architecture
The autoencoder comprises two main components:

<pre>**Encoder:** Compresses the input image into a lower-dimensional latent space.

**Decoder:** Reconstructs the image from the latent representation.

The architecture is defined as follows:

```python import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)   # Latent space representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()  # Output normalized between [-1, 1]
        )

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, 28, 28)  # Reshape back to image dimensions
        return x ```
</pre>

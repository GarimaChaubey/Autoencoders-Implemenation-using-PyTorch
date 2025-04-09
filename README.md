# Autoencoder Implementation with PyTorch

This repository contains an implementation of an autoencoder using PyTorch, trained on the MNIST dataset of handwritten digits. The objective is to learn a compressed latent representation of images and reconstruct them with minimal loss.

## Overview
An autoencoder is an unsupervised neural network that learns to compress input data into a latent space and then reconstruct it back to the original form. This project demonstrates the construction and training of an autoencoder to work with the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits.

![image](https://github.com/user-attachments/assets/1e0fef71-4e14-42b6-b544-a73f05406e1f)

## Model Architecture
The autoencoder comprises two main components:

**Encoder:** Compresses the input image into a lower-dimensional latent space.

**Decoder:** Reconstructs the image from the latent representation.

**The architecture is defined as follows:**

 ```
 python import torch.nn as nn

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
        return x
```


## Dataset
The model is trained and evaluated on the **MNIST** dataset, which is available through torchvision.datasets. The dataset is preprocessed with the following transformations:

**ToTensor**: Converts images to PyTorch tensors.

**Normalize**: Scales pixel values to the range [-1, 1].

```
python import torchvision.transforms as transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = MNIST(root='./kaggle/working', train=True, transform=transform, download=True)
test_dataset = MNIST(root='./kaggle/working', train=False, transform=transform, download=True)
</pre>

## Training
The training process involves minimizing the reconstruction loss between the input images and their reconstructions. The **Mean Squared Error (MSE)** loss function and the **Adam optimizer** are used for this purpose.

<pre>
import torch
from torch.utils.data import DataLoader

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, Loss, Optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

## Evaluation and Visualization
After training, the model's performance can be evaluated by visualizing the original and reconstructed images.

```
python import matplotlib.pyplot as plt

# Load test data
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
dataiter = iter(test_loader)
images, _ = dataiter.next()
images = images.to(device)

# Get reconstructed images
reconstructed = model(images)

# Plot original and reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([images, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.cpu().detach().numpy().reshape((28, 28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
```

## Results
The autoencoder successfully learns to reconstruct handwritten digits with high fidelity. Below are sample results showing original images alongside their reconstructions:



## References
-https://arxiv.org/pdf/1406.2661

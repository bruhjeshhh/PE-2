from generator import Generator
from discriminator import Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Loss function
criterion = nn.BCELoss()

# Training settings
num_epochs = 100
latent_dim = 100
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.ImageFolder(root="/home/bruhjeshh/Coding /PE-2/archive", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Move models to GPU if available
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.shape[0]
        
        # Move to device
        real_imgs = real_imgs.to(device)

        ### 1️⃣ Train Discriminator ###
        optimizer_D.zero_grad()
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Real images loss
        real_preds = discriminator(real_imgs)
        real_loss = criterion(real_preds, real_labels)
        
        # Generate fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        
        # Fake images loss
        fake_preds = discriminator(fake_imgs.detach())
        fake_loss = criterion(fake_preds, fake_labels)
        
        # Total Discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        ### 2️⃣ Train Generator ###
        optimizer_G.zero_grad()

        # Generate new fake images
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        # Fool the discriminator
        fake_preds = discriminator(fake_imgs)
        g_loss = criterion(fake_preds, real_labels)

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # Save sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        vutils.save_image(fake_imgs[:25], f"generated_epoch_{epoch+1}.png", normalize=True)

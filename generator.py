import torch
import torch.nn as nn

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=128):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size * 3),
            nn.Tanh()  # Output is normalized between [-1,1]
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 3, 128, 128)  # Reshape to image format
        return img
    

class Discriminator(nn.Module):
    def __init__(self, img_size=128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        return self.model(img_flat)


# Hyperparameters
latent_dim = 100

# Initialize models
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Test Generator
z = torch.randn(1, latent_dim)  # Random noise
fake_img = generator(z)  # Generate fake image
print("Generated Image Shape:", fake_img.shape)  # Should be [1, 3, 128, 128]


import os

dataset_path = "/home/bruhjeshh/Coding /PE-2/GAN/archive"  # Update this with your actual path

# List the first few files
print("Dataset files:", os.listdir(dataset_path)[:10])
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),          # Convert to Tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] for GANs
])

# Load dataset
dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Check sample batch
images, labels = next(iter(dataloader))
print("Batch shape:", images.shape)

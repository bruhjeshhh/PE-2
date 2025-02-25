import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections.abc import Iterable

# Define transformations (resize & normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = "/home/bruhjeshh/Coding /anothatry/dataset"  # Update this
train_dataset = ImageFolder(root=dataset_path, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Check dataset structure
print(f"Classes: {train_dataset.classes}")
print(f"Number of images: {len(train_dataset)}")

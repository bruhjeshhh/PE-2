import os
from PIL import Image

dataset_path = "/home/bruhjeshh/Coding/PE-2/Resnet50+GradCam/dataset1"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        try:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)  # Try opening the image
            img.verify()  # Verify image integrity
        except Exception as e:
            print(f"Corrupt image found: {img_path} -> {e}")

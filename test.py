import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Check CUDA version
print(torch.backends.cudnn.enabled)  # Should be True

import torch


torch.manual_seed(1)
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version (from PyTorch):", torch.version.cuda)
    print("GPU device:", torch.cuda.get_device_name(0))
else:
    print("No CUDA detected by PyTorch.")

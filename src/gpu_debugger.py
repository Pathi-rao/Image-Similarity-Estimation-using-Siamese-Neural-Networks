import torch

# How many GPUs are there?
print(torch.cuda.device_count())


# Which GPU Is The Current GPU?
print(torch.cuda.current_device())


# Get the name of the current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))


# Is PyTorch using a GPU?
print(torch.cuda.is_available())
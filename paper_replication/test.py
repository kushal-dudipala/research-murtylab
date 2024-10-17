import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.current_device())  # Current GPU being used
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # Name of the current GPU

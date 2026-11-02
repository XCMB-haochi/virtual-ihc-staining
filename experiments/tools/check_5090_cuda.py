"""Fail fast if the project environment cannot execute on the intended GPU."""
import torch
assert torch.cuda.is_available()
print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(),flush=True)
x=torch.ones(32,device='cuda')
assert (x*x).sum().item()==32

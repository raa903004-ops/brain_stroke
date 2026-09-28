import torch
import time

device = "cuda"

x = torch.randn(
    5000,
    5000
).to(device)


start=time.time()

y = x @ x

torch.cuda.synchronize()

print(time.time()-start)
print(y.device)
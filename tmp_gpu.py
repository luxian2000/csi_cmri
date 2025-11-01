import torch

device = torch.device("cuda")
x = torch.randn(3,3).to(device)

model = torch.nn.Linear(3,3).to(device)
print(model)

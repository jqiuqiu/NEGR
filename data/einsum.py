import torch

A = torch.randn(3, 4)
B = torch.randn(3, 4, 5)
C=torch.einsum('ijk,jl->il', B, A)
print(C)
import torch
import einops

example = torch.randn(3, 5)
print(example)

example = einops.repeat(example, 'b c -> b t c', t=3)
print(example)

example = example.reshape(-1, 5)
print(example)
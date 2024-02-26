import torch
import torch.nn as nn

a = nn.Embedding(256, 128)
b = nn.Embedding(128, 128)

c = torch.cat([a.weight,b.weight,b.weight],dim=0)
from transformers import AutoModel
import torch.nn as nn
import torch

class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear3 = nn.Sequential(
          nn.Linear(768, 2),
        )

    def forward(self, x):
        xx = x.squeeze(1)
        x_2 = self.linear3(xx)
        x_3 = torch.mean(x_2, dim=0)

        return x_3
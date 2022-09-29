from transformers import AutoModel
import torch.nn as nn
import torch

class MeanAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear3 = nn.Sequential(
          nn.Linear(768, 768),
          nn.Sigmoid(),
        #   nn.Linear(768, 768),
        #   nn.ReLU(),
          nn.Linear(768, 2)
        )

    def forward(self, x):
        x_2 = self.linear3(x)
        # x_22 = self.act_fn(x_2)
        x_3 = torch.mean(x_2, dim=1)
        return x_3
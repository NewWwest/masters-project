from transformers import AutoModel
import torch.nn as nn

class BertAndLinear(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(base_model)
        self.linear1 = nn.Linear(768, 2)
        # self.act_fn = nn.Softmax()
        # self.act_fn = nn.Tanh()

    def forward(self, x):
        x_1 = self.codebert(x)
        x_2 = x_1[0]
        x_22 = x_2[:,0,:]
        x_3 = self.linear1(x_22)
        # x_4 = self.act_fn(x_3)
        return x_3
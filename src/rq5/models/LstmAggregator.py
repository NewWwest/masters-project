from transformers import AutoModel
import torch.nn as nn

class LstmAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768,
            hidden_size=512,
            num_layers =5)
        self.linear1 = nn.Linear(512, 2)
        self.act_fn = nn.Softmax()

    def forward(self, x):
        lenx = x.shape[1]
        out, hidden = self.lstm(x.view(lenx, 1, -1))
        x_3 = self.linear1(out[-1])
        # x_3 = self.linear1(hidden[0].squeeze(dim=0))
        # x_4 = self.act_fn(x_3)
        return x_3
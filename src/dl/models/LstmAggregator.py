from unicodedata import bidirectional
from transformers import AutoModel
import torch.nn as nn

class LstmAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768,
            hidden_size=512,
            num_layers = 5,
            # bidirectional=True,
            dropout=0.2,
            batch_first = False)
        self.linear1 = nn.Linear(512, 2)
        self.act_fn = nn.Softmax()

    def forward(self, x):
        xx = x.squeeze(1)
        xx = xx.squeeze(1)
        lenx = xx.shape[0]
        out = self.lstm(xx.view(lenx, 1, -1))
        x_3 = self.linear1(out[0][-1])
        return x_3
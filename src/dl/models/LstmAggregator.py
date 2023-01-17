# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
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
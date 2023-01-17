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
from transformers import AutoModel
import torch.nn as nn

class FrozenBertAndLinear(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(base_model)
        self.linear1 = nn.Linear(768, 2)
        self.act_fn = nn.Tanh()

        for param in self.codebert.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_1 = self.codebert(x)
        x_2 = x_1[0]
        x_22 = x_2[:,0,:]
        x_3 = self.linear1(x_22)
        x_4 = self.act_fn(x_3)
        return x_4
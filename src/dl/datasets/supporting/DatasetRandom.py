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
import torch.utils.data as data
import torch

class DatasetRandom(data.Dataset):
    def __init__(self, size_of_vector, size, max_value = 1023):
        super().__init__()
        self.size_of_vector = size_of_vector
        self.size = size
        self.max_value = max_value
        self.gen_data()

    def gen_data(self):
        data = torch.randint(low=0, high=self.max_value, size=(self.size, self.size_of_vector))
        label = [1 if torch.sum(x) > 100_000 else 0 for x in data]

        data[:,0] = 0
        self.data = data
        self.label = torch.tensor([[x, 1-x] for x in label])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
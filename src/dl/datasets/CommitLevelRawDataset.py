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
from src.dl.datasets.BaseRawDataset import BaseRawDataset
import torch


class CommitLevelRawDataset(BaseRawDataset):
    def __init__(self):
        super().__init__()


    def _load_file(self, collection, json_file, commit_id):
        data = [x['commit_sample'] for x in json_file
            if 'commit_sample' in x and x['commit_sample'] != None and len(x['commit_sample']) > 0]

        if len(data) > 0:  
            collection.append((commit_id, [torch.Tensor(x) for x in data]))
            return 1
        else:
            return 0

    

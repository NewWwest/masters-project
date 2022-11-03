from src.rq5.datasets.BaseRawDataset import BaseRawDataset
import torch
import numpy as np


class SampleLevelRawDataset(BaseRawDataset):
    def __init__(self):
        # super().__init__('positive-encodings', 'background-encodings')
        super().__init__()


    def _load_file(self, collection, json_file):
        data = [x['commit_sample'] for x in json_file
            if 'commit_sample' in x and x['commit_sample'] != None and len(x['commit_sample']) > 0]
            
        if len(data) > 0:
            tensors = torch.stack([torch.Tensor(x).int() for x in data])
            tensors = torch.unique(tensors, dim=0)
            collection += tensors

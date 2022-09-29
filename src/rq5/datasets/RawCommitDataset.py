import json
import torch
import torch.utils.data as data
from src.utils.utils import get_files_in_from_directory


class RawCommitDataset(data.Dataset):
    def __init__(self, path_to_files):
        self.positive_data = []
        self.background_data = []
        self._load_files(path_to_files)
        

    def _load_files(self, path):
        positive_json_files = get_files_in_from_directory(path, extension='.pt', startswith='batch-positive')
        background_json_files = get_files_in_from_directory(path, extension='.pt', startswith='batch-background')

        for filename in positive_json_files:
            t = torch.load(filename)
            self.positive_data.append(t)

        for filename in background_json_files:
            t = torch.load(filename)
            self.background_data.append(t)


    def __len__(self):
        return len(self.positive_data) + len(self.background_data) 


    def __getitem__(self, idx):
        if idx < len(self.positive_data):
            data_point = self.positive_data[idx]
            data_label = torch.Tensor([1, 0]) 
        else:
            data_point = self.background_data[idx-len(self.positive_data)]
            data_label = torch.Tensor([0, 1]) 
        
        return data_point, data_label
import json
import torch
import torch.utils.data as data
from src.utils.utils import get_files_in_from_directory


class RawDataset(data.Dataset):
    def __init__(self, path_to_files):
        self.positive_data = []
        self.background_data = []
        self.data = []
        self.labels = []
        
        self._load_files(path_to_files)
        

    def _load_files(self, path):
        positive_json_files = get_files_in_from_directory(path, extension='.json', startswith='batch-positive')
        background_json_files = get_files_in_from_directory(path, extension='.json', startswith='batch-background')

        for filename in positive_json_files:
            with open(filename, 'r') as f:
                temp_data = json.load(f)
                self.positive_data += temp_data

        for filename in background_json_files:
            with open(filename, 'r') as f:
                temp_data = json.load(f)
                self.background_data += temp_data



        self.data = torch.Tensor(self.positive_data + self.background_data).int()
        # self.labels = torch.Tensor([[1, -1] for x in range(len(self.positive_data))] + [[-1, 1] for x in range(len(self.background_data))]).int()
        self.labels = torch.Tensor([[1, 0] for x in range(len(self.positive_data))] + [[0, 1] for x in range(len(self.background_data))]).int()

    
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label
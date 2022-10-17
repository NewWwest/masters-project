from abc import abstractmethod
import json
import torch
import torch.utils.data as data
import random


from src.utils.utils import get_files_in_from_directory


class BaseRawDataset(data.Dataset):
    def __init__(self, positive_prefix, background_prefix):
        self.positive_prefix = positive_prefix
        self.background_prefix = background_prefix
        self.positive_data = []
        self.background_data = []


    @abstractmethod
    def _load_file(self, json_file):
        pass


    def reject_data(self, fraction):
        self.positive_data = random.sample(self.positive_data, int(fraction*len(self.positive_data)))
        self.background_data = random.sample(self.background_data, int(fraction*len(self.background_data)))

        
    def split_data(self, fraction):
        positive_cut_point = int(fraction*len(self.positive_data))
        background_cut_point = int(fraction*len(self.background_data))

        random.shuffle(self.positive_data)
        random.shuffle(self.background_data)
        part_a = BaseRawDataset(self.positive_prefix, self.background_prefix)
        part_a.positive_data = self.positive_data[:positive_cut_point]
        part_a.background_data = self.background_data[:background_cut_point]

        part_b = BaseRawDataset(self.positive_prefix, self.background_prefix)
        part_b.positive_data = self.positive_data[positive_cut_point:]
        part_b.background_data = self.background_data[background_cut_point:]

        return part_a, part_b


    def load(self, path):
        positive_json_files = get_files_in_from_directory(path, extension='.json', startswith=self.positive_prefix)
        background_json_files = get_files_in_from_directory(path, extension='.json', startswith=self.background_prefix)

        positive_data_temp = []
        background_data_temp = []

        for filename in positive_json_files:
            with open(filename, 'r') as f:
                temp_data = json.load(f)
                self._load_file(positive_data_temp, temp_data)

        for filename in background_json_files:
            with open(filename, 'r') as f:
                temp_data = json.load(f)
                self._load_file(background_data_temp, temp_data)

        self.positive_data = positive_data_temp
        self.background_data = background_data_temp
    
    def __len__(self):
        return len(self.positive_data) + len(self.background_data)


    def __getitem__(self, idx):
        if idx < len(self.positive_data):
            data_point = self.positive_data[idx]
            data_label = torch.Tensor([1, 0]) 
        else:
            data_point = self.background_data[idx - len(self.positive_data)]
            data_label = torch.Tensor([0, 1]) 
        
        return data_point, data_label
        
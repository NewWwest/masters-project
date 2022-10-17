from src.rq5.datasets.BaseRawDataset import BaseRawDataset
from math import floor
import torch.utils.data as data
import torch
import random

class OverSampledDataset(data.Dataset):
    def __init__(self, base_dataset: BaseRawDataset, ratio):
        super().__init__()
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.data = []
        self.labels = []

        self._oversample_to_ratio()
        

    def _oversample_to_ratio(self):
        self.data = []
        self.labels = []

        target_number_of_positive = int(len(self.base_dataset.background_data) / self.ratio)
        whole_repetition = floor(target_number_of_positive/len(self.base_dataset.positive_data))
        self.data = self.base_dataset.positive_data * whole_repetition
        extra_sampled = random.sample(self.base_dataset.positive_data, target_number_of_positive-len(self.data))
        self.data += extra_sampled

        self.labels = [[1, 0] for _ in self.data] + [[0, 1] for _ in self.base_dataset.background_data]
        # self.labels = [[1, -1] for _ in self.data] + [[-1, 1] for _ in background_data]
        self.data += self.base_dataset.background_data
        

        self.labels = torch.Tensor(self.labels)
        self.labels = self.labels.int()
        print('Data loaded')

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label
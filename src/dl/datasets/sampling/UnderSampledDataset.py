from src.dl.datasets.BaseRawDataset import BaseRawDataset
import torch.utils.data as data
import torch
import random

class UnderSampledDataset(data.Dataset):
    def __init__(self, base_dataset:BaseRawDataset, ratio):
        super().__init__()
        self.data = []
        self.labels = []
        self.base_dataset = base_dataset

        self._undersample_to_ratio(ratio)
        

    def _undersample_to_ratio(self, ratio):
        target_number_of_background = int(len(self.base_dataset.positive_data) * ratio)
        if target_number_of_background > len(self.base_dataset.background_data):
            raise Exception("Cannot undersample")

        background_sampled = random.sample(self.base_dataset.background_data, target_number_of_background)
        self.data = self.base_dataset.positive_data + background_sampled

        # self.labels = [torch.Tensor([1, -1]).int() for x in positive_data] + [torch.Tensor([-1, 1]).int() for x in background_sampled]
        self.labels = [torch.Tensor([1, 0]).int() for x in self.base_dataset.positive_data] + [torch.Tensor([0, 1]).int() for x in background_sampled]
        print('Data loaded')

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label
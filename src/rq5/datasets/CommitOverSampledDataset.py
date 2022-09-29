from math import floor
import torch.utils.data as data
import torch
import random

class CommitOverSampledDataset(data.Dataset):
    def __init__(self, base_dataset: data.Dataset, ratio):
        super().__init__()
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.positive_data = []
        self.background_data = []

        self._oversample_to_ratio()
        

    def _oversample_to_ratio(self):
        all_background_data = []
        all_positive_data = []

        for x in range(len(self.base_dataset)):
            data_point = self.base_dataset[x]
            if data_point[1][0] == 1:
                all_positive_data.append(data_point[0])
            else:
                all_background_data.append(data_point[0])


        self.positive_data = []
        self.background_data = all_background_data

        target_number_of_positive = int(len(all_background_data) / self.ratio)
        whole_repetition = floor(target_number_of_positive/len(all_positive_data))
        self.positive_data = all_positive_data * whole_repetition
        extra_sampled = random.sample(all_positive_data, target_number_of_positive-len(self.positive_data))
        self.positive_data += extra_sampled

        self.length = len(self.positive_data) + len(self.background_data)
        print('Data loaded')

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if idx < len(self.positive_data):
            data_point = self.positive_data[idx]
            data_label = torch.Tensor([1, 0]) 
        else:
            data_point = self.background_data[idx-len(self.positive_data)]
            data_label = torch.Tensor([0, 1]) 
        
        return data_point, data_label
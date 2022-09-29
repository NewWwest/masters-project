from math import floor
import torch.utils.data as data
import torch
import random

class OverSampledDataset(data.Dataset):
    def __init__(self, base_dataset: data.Dataset, ratio):
        super().__init__()
        self.base_dataset = base_dataset
        self.ratio = ratio
        self.data = []
        self.labels = []

        self._oversample_to_ratio()
        

    def _oversample_to_ratio(self):
        background_data = []
        positive_data = []

        for x in range(len(self.base_dataset)):
            data_point = self.base_dataset[x]
            if data_point[1][0] == 1:
                positive_data.append(data_point[0])
            else:
                background_data.append(data_point[0])


        self.data = []
        self.labels = []

        target_number_of_positive = int(len(background_data) / self.ratio)
        whole_repetition = floor(target_number_of_positive/len(positive_data))
        self.data = positive_data * whole_repetition
        extra_sampled = random.sample(positive_data, target_number_of_positive-len(self.data))
        self.data += extra_sampled

        self.labels = [[1, 0] for _ in self.data] + [[0, 1] for _ in background_data]
        # self.labels = [[1, -1] for _ in self.data] + [[-1, 1] for _ in background_data]
        self.data += background_data
        

        self.labels = torch.Tensor(self.labels)
        self.labels = self.labels.int()
        self.data = torch.stack(self.data)
        print('Data loaded')

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label
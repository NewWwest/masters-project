import torch.utils.data as data
import torch
import random

class UnderSampledDataset(data.Dataset):
    def __init__(self, base_dataset, ratio):
        super().__init__()
        self.data = []
        self.labels = []

        self._undersample_to_ratio(base_dataset, ratio)
        

    def _undersample_to_ratio(self, base_dataset, ratio):
        background_data = []
        positive_data = []

        for x in range(len(base_dataset)):
            data_point = base_dataset[x]
            if data_point[1][0] == 1:
                positive_data.append(data_point[0])
            else:
                background_data.append(data_point[0])

        target_number_of_background = int(len(positive_data) * ratio)
        if target_number_of_background > len(background_data):
            raise Exception("Cannot undersample")

        background_sampled = random.sample(background_data, target_number_of_background)
        self.data = positive_data + background_sampled
        # self.labels = [torch.Tensor([1, -1]).int() for x in positive_data] + [torch.Tensor([-1, 1]).int() for x in background_sampled]
        self.labels = [torch.Tensor([1, 0]).int() for x in positive_data] + [torch.Tensor([0, 1]).int() for x in background_sampled]
        # self.labels = [[1, -1] for _ in self.data] + [[-1, 1] for _ in background_data]
        print('Data loaded')

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.labels[idx]
        return data_point, data_label
import torch.utils.data as data
import torch

class CsvDataset(data.Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.size = len(data)
        self.data = torch.tensor(data).int()
        self.label = torch.tensor(labels).int()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
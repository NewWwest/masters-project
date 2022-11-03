from abc import abstractmethod
import json
import torch
import torch.utils.data as data
import random

# ['js', 'jsx', 'ts', 'tsx', ]
VALID_EXTENSIONS = set(['java'])

class BaseRawDataset(data.Dataset):
    def __init__(self):
        self.positive_data = []
        self.background_data = []


    @abstractmethod
    def _load_file(self, collection, json_file):
        pass


    def crop_data(self, positive_count, background_count):
        self.positive_data = random.sample(self.positive_data, min(positive_count, len(self.positive_data)))
        self.background_data = random.sample(self.background_data, min(background_count, len(self.background_data)))


    def limit_data(self, limit):
        final_positive_count = int(limit*len(self.positive_data)/(len(self.positive_data)+len(self.background_data)))
        self.positive_data = random.sample(self.positive_data, min(final_positive_count, len(self.positive_data)))
        self.background_data = random.sample(self.background_data, min(limit-final_positive_count, len(self.background_data)))
        

    def setup_ratios(self, oversampling_ratio, class_ratio, samples_limit):
        if oversampling_ratio == -1:
            oversampling_ratio = int(len(self.background_data)/(class_ratio*len(self.positive_data)))

        positive_count = len(self.positive_data) * oversampling_ratio
        self.positive_data = self.positive_data * oversampling_ratio
        background_count = class_ratio*len(self.positive_data)
        self.background_data = random.sample(self.background_data, min(background_count, len(self.background_data)))

        final_positive_count = int(samples_limit*positive_count/(positive_count+background_count))
        self.positive_data = random.sample(self.positive_data, min(final_positive_count, len(self.positive_data)))
        self.background_data = random.sample(self.background_data, min(samples_limit-final_positive_count, len(self.background_data)))



        
    def split_data(self, fraction):
        positive_cut_point = int(fraction*len(self.positive_data))
        background_cut_point = int(fraction*len(self.background_data))

        random.shuffle(self.positive_data)
        random.shuffle(self.background_data)
        part_a = BaseRawDataset()
        part_a.positive_data = self.positive_data[:positive_cut_point]
        part_a.background_data = self.background_data[:background_cut_point]

        part_b = BaseRawDataset()
        part_b.positive_data = self.positive_data[positive_cut_point:]
        part_b.background_data = self.background_data[background_cut_point:]

        return part_a, part_b


    def load_files(self, positive_json_files, background_json_files):
        positive_data_temp = []
        background_data_temp = []

        for filename in positive_json_files:
            try:
                with open(filename, 'r') as f:
                    temp_data = json.load(f)
                    temp_data = [x for x in temp_data if x['file_name'].split('.')[-1] in VALID_EXTENSIONS]
                    self._load_file(positive_data_temp, temp_data)
            except Exception as e:
                print('Failed to load', filename)
                print(e)

        for filename in background_json_files:
            try:
                with open(filename, 'r') as f:
                    temp_data = json.load(f)
                    temp_data = [x for x in temp_data if x['file_name'].split('.')[-1] in VALID_EXTENSIONS]
                    self._load_file(background_data_temp, temp_data)
            except Exception as e:
                print('Failed to load', filename)
                print(e)

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
        
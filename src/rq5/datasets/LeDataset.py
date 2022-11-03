from abc import abstractmethod
import json
import torch
import torch.utils.data as data
import random
from src.rq5.datasets.BaseRawDataset import BaseRawDataset

from src.rq5.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.rq5.datasets.CommitLevelRawDataset import CommitLevelRawDataset

from src.utils.utils import get_files_in_from_directory
import torch

# ['js', 'jsx', 'ts', 'tsx', ]
VALID_EXTENSIONS = set(['java'])


def load_commit_level(path):

    train_dataset = CommitLevelRawDataset()
    train_dataset.load_files(positive_json_files, background_json_files)
    test_dataset = CommitLevelRawDataset()
    test_dataset.load_files(positive_test, background_test)

    return train_dataset, test_dataset


    
def _select_repos_for_test(files, fraction = 0.15):
    repos = set()
    for x in files:
        segments = x.split('-')
        repo_semi_full_name = f'{segments[2]}-{segments[3]}'
        repos.add(repo_semi_full_name)

    test_set = random.sample(repos, int(fraction*len(repos)))
    return test_set
        
   
def _test_train_split(filenames, test_repos):
    new_positive_json_files = []
    test_positive_json_files = []
    for x in filenames:
        is_test = False
        for r in test_repos:
            if r in x:
                is_test = True
                break
        if is_test:
            test_positive_json_files.append(x)
        else:
            new_positive_json_files.append(x)

    return new_positive_json_files, test_positive_json_files




class OmniDataset():
    def __init__(self, path):
        positive_json_files = get_files_in_from_directory(path, extension='.json', startswith='embedded-positive-encodings')
        background_json_files = get_files_in_from_directory(path, extension='.json', startswith='embedded-background-encodings')
        repos_test_set = _select_repos_for_test(positive_json_files)

        positive_json_files, positive_test = _test_train_split(positive_json_files, repos_test_set)
        background_json_files, background_test = _test_train_split(background_json_files, repos_test_set)
        self.positive_data = []
        self.background_data = []


    def _load_file(self, collection, json_file):
        data = [x['commit_sample'] for x in json_file
            if 'commit_sample' in x and x['commit_sample'] != None and len(x['commit_sample']) > 0]

        if len(data) > 0:  
            collection.append([torch.Tensor(x) for x in data])


    def crop_data(self, positive_count, background_count):
        self.positive_data = random.sample(self.positive_data, min(positive_count, len(self.positive_data)))
        self.background_data = random.sample(self.background_data, min(background_count, len(self.background_data)))

        
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


    def return_sample_level(self):
        pass



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
        
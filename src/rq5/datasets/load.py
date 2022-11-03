from abc import abstractmethod
import os
import json
import torch
import torch.utils.data as data
import random
import tarfile

from src.rq5.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.rq5.datasets.CommitLevelRawDataset import CommitLevelRawDataset

from src.utils.utils import get_files_in_from_directory


def load_sample_level(path):

    train_dataset = SampleLevelRawDataset()
    train_dataset.load_files(positive_json_files, background_json_files)
    test_dataset = SampleLevelRawDataset()
    test_dataset.load_files(positive_test, background_test)

    return train_dataset, test_dataset


def load_commit_level(path):
    positive_json_files = get_files_in_from_directory(path, extension='.json', startswith='embedded-positive-encodings')
    background_json_files = get_files_in_from_directory(path, extension='.json', startswith='embedded-background-encodings')
    repos_test_set = _select_repos_for_test(positive_json_files)

    positive_json_files, positive_test = _test_train_split(positive_json_files, repos_test_set)
    background_json_files, background_test = _test_train_split(background_json_files, repos_test_set)

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

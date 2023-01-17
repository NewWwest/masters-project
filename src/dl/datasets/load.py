# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
import random

from src.dl.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.dl.datasets.CommitLevelRawDataset import CommitLevelRawDataset

from src.utils.utils import get_files_in_from_directory


def load_sample_level(path):
    train_fraction = 0.8
    positive_json_files = get_files_in_from_directory(path, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(path, extension='.json', startswith='background-encodings')

    random.shuffle(positive_json_files)
    random.shuffle(background_json_files)

    positive_train_size = int(train_fraction * len(positive_json_files))
    positive_train = positive_json_files[:positive_train_size]
    positive_test = positive_json_files[positive_train_size:]

    background_train_size = int(train_fraction * len(background_json_files))
    background_train = background_json_files[:background_train_size]
    background_test = background_json_files[background_train_size:]

    train_dataset = SampleLevelRawDataset()
    train_dataset.load_files(positive_train, background_train)
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

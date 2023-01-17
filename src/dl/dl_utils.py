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
import json
import pandas as pd
import random
import os

from src.dl.datasets.supporting.CsvDataset import CsvDataset

def save_dataset(dataset, filename):
    data = []
    for x, l in dataset:
        if x.__class__.__name__ == 'list':
            features = x
        else:
            features = x.tolist()
        data.append({'label':json.dumps(l.tolist()), 'data':json.dumps(x)})

    pd.DataFrame(data).to_csv(filename, index=False)

def read_dataset(filename):
    df = pd.read_csv(filename)
    return CsvDataset([json.loads(x) for x in df['data']], [json.loads(x) for x in df['label']])

    
def get_repo_seminames(files):
    repos = set()
    for x in files:
        segments = x.split('-')
        repo_semi_full_name = f'{segments[2]}-{segments[3]}'
        repos.add(repo_semi_full_name)

    return repos


def get_files_in_set(filenames, test_repos):
    filtered_json_files = []
    for x in filenames:
        is_test = False
        for r in test_repos:
            if r in x:
                is_test = True
                break
        if is_test:
            filtered_json_files.append(x)

    return filtered_json_files


def chunks(array, number_of_chunks):
    for i in range(0, number_of_chunks):
        yield array[i::number_of_chunks]

        
def save_file_datasets(file_dataset, dataset_type, dir):
    data = {
        'positive_files': file_dataset[0],
        'background_files': file_dataset[1]
    }
    with open(os.path.join(dir, f'{dataset_type}-files.json'), 'w') as f:
        json.dump(data, f)


def load_file_dataset(dataset_type, dir):
    with open(os.path.join(dir, f'{dataset_type}-files.json'), 'r') as f:
        data = json.load(f)

    return (data['positive_files'], data['background_files'])
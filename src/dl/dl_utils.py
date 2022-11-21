import json
import pandas as pd
import random

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

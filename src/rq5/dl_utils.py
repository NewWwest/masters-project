import json
import pandas as pd

from src.rq5.datasets.CsvDataset import CsvDataset

def save_dataset(dataset, filename):
    data = []
    for x, l in dataset:
        data.append({'label':json.dumps(l.tolist()), 'data':json.dumps(x.tolist())})

    pd.DataFrame(data).to_csv(filename, index=False)

def read_dataset(filename):
    df = pd.read_csv(filename)
    return CsvDataset([json.loads(x) for x in df['data']], [json.loads(x) for x in df['label']])

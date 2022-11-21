from math import ceil
import sys
sys.path.insert(0, r'D:\Projects\aaa')

import random
random.seed(42)

from transformers import AutoModel
import os
import torch
import json
import time
import json
from multiprocessing import Pool
import tqdm
from src.dl.models.BertAndLinear import BertAndLinear
from src.utils.utils import get_files_in_from_directory


base_model = 'microsoft/graphcodebert-base'
VALID_EXTENSIONS = set(['java'])

# technical configuration
data_location = r'results/dl/java/RollingWindowMiner' 
results_dir = r'results/dl/java/RollingWindowMiner_embedded1' 
max_samples_per_file = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cpus = 6



def get_tokenizer():
    model = BertAndLinear(base_model)
    model.load_state_dict(torch.load(r'D:\Projects\aaa\src\rq5\binaries\t3_roll_java\model_t3_roll_java_final.pickle'))
    transformer = model.codebert

    # transformer = AutoModel.from_pretrained(base_model)
    
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()
    transformer.to(device)

    return transformer


def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

def tokenize_samples(data_files):
    tokenizer = get_tokenizer()
    for data_file in data_files:
        print(data_file)
        with open(data_file, 'r') as f:
            data = json.load(f)

        embeddings = []
        for data_point in data:
            if 'commit_sample' in data_point and \
                data_point['commit_sample'] != None and \
                len(data_point['commit_sample']) > 0 and \
                data_point['file_name'].split('.')[-1] in VALID_EXTENSIONS:

                tensor = torch.Tensor(data_point['commit_sample']).int()
                tensor = tensor[None, :] # Extend to a batch mode
                tensor = tensor.to(device)
                result = tokenizer(tensor)
                labels = result[0][:,0,:]
                labels_in_memory = labels.cpu()
                res = {
                    'commit_id': data_point['commit_id'],
                    'file_name': data_point['file_name'],
                    'is_security_related': data_point['is_security_related'],
                    'commit_sample': labels_in_memory.tolist()
                }
                embeddings.append(res)

        if len(embeddings) > 0:
            file_name = os.path.basename(data_file)
            new_file = os.path.join(results_dir, 'embedded-' + file_name)
            with open(new_file, 'w') as f:
                json.dump(embeddings, f)
    



def main():
    positive_json_files = get_files_in_from_directory(data_location, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(data_location, extension='.json', startswith='background-encodings')
    data_files = positive_json_files + background_json_files
    group_data_files = split(data_files, 1000)

    with Pool(cpus) as p:
        with tqdm.tqdm(total=ceil(len(data_files)/1000)) as pbar:
            for _ in p.imap_unordered(tokenize_samples, group_data_files, chunksize=1):
                pbar.update()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
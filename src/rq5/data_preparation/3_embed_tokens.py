import sys
sys.path.insert(0, r'D:\Projects\aaa')

import random
random.seed(42)

from transformers import AutoModel
import os
import torch
import json
import time
import unicodedata
import regex as re
import json
from multiprocessing import Pool
import tqdm
import uuid
from transformers import AutoTokenizer

from src.rq5.models.BertAndLinear import BertAndLinear

from src.utils.utils import get_files_in_from_directory

# technical configuration
batch_size_ = 64
data_location = r'results/tokenized_rolling_window_miner_results_eval'
output_directory = r'results/embedded_rolling_window_miner_results_eval'
# data_location = r'results/tokenized_rolling_window_miner_results_train'
# output_directory = r'results/embedded_rolling_window_miner_results_train'
max_samples_per_file = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_tokenizer():
    model = BertAndLinear()
    model.load_state_dict(torch.load(r'src\rq5\binaries\rolling_window_npm_cre_0\model_rolling_window_npm_cre_0_epoch_5.pickle'))
    transformer = model.codebert

    # transformer = AutoModel.from_pretrained("microsoft/codebert-base")
    
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()
    transformer.to(device)

    return transformer

def chunks(iter, max_chunk_size):
    for i in range(0, len(iter), max_chunk_size):
        yield iter[i:i + max_chunk_size]


def tokenize_samples(data_file, tokenizer):
    file_name = os.path.basename(data_file)
    with open(data_file, 'r') as f:
        data = json.load(f)

    embeddings = []
    for vector in chunks(data, batch_size_):
        tensor = torch.Tensor(vector).int()
        tensor = tensor.to(device)
        result = tokenizer(tensor)[0][:,0,:]
        labels_in_memory = result.cpu()
        embeddings += labels_in_memory

    asd = torch.stack(embeddings)
    torch.save(asd, f'{output_directory}/{file_name}.pt')


def main():
    tokenizer = get_tokenizer()
    data_files = get_files_in_from_directory(data_location, extension='.json')
    data_files = random.sample(data_files, int(len(data_files)/3))
    
    with tqdm.tqdm(total=len(data_files)) as pbar:
        for f in data_files:
            tokenize_samples(f, tokenizer)
            pbar.update()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
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
from src.rq5.models.BertAndLinear import BertAndLinear
from src.utils.utils import get_files_in_from_directory


base_model = 'microsoft/graphcodebert-base'

# technical configuration
data_location = r'results/new_mined_code_added_code'
max_samples_per_file = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_tokenizer():
    model = BertAndLinear(base_model)
    model.load_state_dict(torch.load(r'D:\Projects\aaa\src\rq5\binaries\new_added_code_mined_test\model_new_added_code_mined_test_epoch_0.pickle'))
    transformer = model.codebert

    # transformer = AutoModel.from_pretrained(base_model)
    
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.eval()
    transformer.to(device)

    return transformer


def tokenize_samples(data_file, tokenizer):
    with open(data_file, 'r') as f:
        data = json.load(f)

    embeddings = []
    for data_point in data:
        if 'commit_sample' in data_point and data_point['commit_sample'] != None and len(data_point['commit_sample']) > 0:
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

    file_name = os.path.basename(data_file)
    new_file = os.path.join(os.path.dirname(data_file), 'embedded-' + file_name)
    with open(new_file, 'w') as f:
        json.dump(embeddings, f)
    



def main():
    tokenizer = get_tokenizer()
    positive_json_files = get_files_in_from_directory(data_location, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(data_location, extension='.json', startswith='background-encodings')
    data_files = positive_json_files + background_json_files
    
    with tqdm.tqdm(total=len(data_files)) as pbar:
        for f in data_files:
            tokenize_samples(f, tokenizer)
            pbar.update()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
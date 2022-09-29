import sys
sys.path.insert(0, r'D:\Projects\aaa')

import json
import time
import unicodedata
import regex as re
import json
from multiprocessing import Pool
import tqdm
import uuid
from transformers import AutoTokenizer

from src.utils.utils import get_files_in_from_directory

# technical configuration
cpus = 8
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
data_location = 'results/rolling_window_miner_results_train'
output_directory = r'results/tokenized_rolling_window_miner_results_train'
max_samples_per_file = 1024

# extensions to include in token mining
valid_extensions = set()
# valid_extensions.update(['java', 'scala', 'kt', 'swift']) #27k
valid_extensions.update(['js', 'jsx', 'ts']) #67k
# valid_extensions.update(['py', 'ipynb']) #29k
# valid_extensions.update(['php']) # 100k
# valid_extensions.update(['cpp', 'c', 'cs', 'cshtml', 'sql', 'r', 'vb']) #142k

# tokenization options
splitter_regex = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|((?<![a-zA-Z0-9])(?=[a-zA-Z0-9])|(?<=[a-zA-Z0-9])(?![a-zA-Z0-9]))')
model_input_size = 512
max_tokenizer_input = 128*1024 #for RAM paging issues 


def tokenize_samples(data_file):
    process_id = str(uuid.uuid4())
    samples_count = 0

    with open(data_file, 'r') as f:
        data = json.load(f)

    for commit_hash in data:
        positive_samples = []
        background_samples = []
        for datapoint in data[commit_hash]:
            try:
                # commit_id = datapoint['commit_id']
                is_security_related = datapoint['is_security_related']
                commit_title = datapoint['commit_title']
                sample = datapoint['commit_sample']
                file_name = datapoint['file_name']

                if file_name.split('.')[-1] not in valid_extensions:
                    continue
                if sample == None or len(sample) == 0:
                    continue

                message_tokens = tokenizer.tokenize(commit_title)

                normalized_tokens = splitter_regex.split(str(unicodedata.normalize('NFKD', sample).encode('ascii', 'ignore')))
                normalized_tokens = [t for t in normalized_tokens if t]
                to_tokenize = ' '.join(normalized_tokens)
                to_tokenize = to_tokenize[0:min(len(to_tokenize), max_tokenizer_input)]

                code_tokens = tokenizer.tokenize(to_tokenize)
                code_tokens_max_size = model_input_size - 3 - len(message_tokens)
                code_tokens = code_tokens[0:min(code_tokens_max_size, len(code_tokens))]

                tokens = [tokenizer.cls_token] + message_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
                if len(tokens) < model_input_size:
                    tokens = tokens + [tokenizer.pad_token] * (model_input_size-len(tokens))

                if is_security_related:
                    positive_samples.append(tokenizer.convert_tokens_to_ids(tokens))
                else:
                    background_samples.append(tokenizer.convert_tokens_to_ids(tokens))
            except Exception as e:
                print('Exception')
                print(e)  
                
        if len(positive_samples) > 0:
            samples_count += len(positive_samples)
            with open(f'{output_directory}/batch-positive-{process_id}-{commit_hash}.json', 'w') as f:
                json.dump(positive_samples, f)

        if len(background_samples) > 0:
            samples_count += len(positive_samples)
            with open(f'{output_directory}/batch-background-{process_id}-{commit_hash}.json', 'w') as f:
                json.dump(background_samples, f)


    return samples_count

def main():
    data_files = get_files_in_from_directory(data_location, extension='.json')
    with Pool(cpus) as p:
        with tqdm.tqdm(total=len(data_files)) as pbar:
            for samples_count in p.imap_unordered(tokenize_samples, data_files, chunksize=1):
                pbar.update()



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
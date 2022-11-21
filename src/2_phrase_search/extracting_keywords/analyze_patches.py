import json
import pandas as pd
import time
import unicodedata
import regex as re
import json
from multiprocessing import Pool
import tqdm
import uuid

from src.utils.utils import get_files_in_from_directory

# technical configuration
cpus = 20
batch_size = 1024

data_location = 'location_to_diff_files'
positive_directory = 'directory_of_positive_samples'
background_directory = 'directory_of_background_samples'
token_ranking_file = 'result_file.csv'

# extensions to include in token mining
valid_extensions = set()
valid_extensions.update(['java', 'scala', 'kt', 'swift'])
valid_extensions.update(['js', 'jsx', 'ts'])
valid_extensions.update(['py', 'ipynb'])
valid_extensions.update(['cpp', 'c', 'cs', 'cshtml', 'sql', 'r', 'vb', 'php'])

# tokenization options
splitter_regex = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|((?<![a-zA-Z0-9])(?=[a-zA-Z0-9])|(?<=[a-zA-Z0-9])(?![a-zA-Z0-9]))')
filter_regex = re.compile(r'[a-zA-Z0-10]{4,20}')
minimal_token_size = 4
maximal_token_size = 20

# token ranking options
minimal_commit_count = 5
minimal_recall = 0.95


def pre_tokenize_samples(data_file):
    process_id = str(uuid.uuid4())
    positive_samples = []
    saved_positive_samples = 0
    background_samples = []
    saved_background_samples = 0

    with open(data_file, 'r') as f:
        data = json.load(f)

    for hash in data:
        try:
            samples = []
            is_security_related = data[hash]['is_security_related']
            for file in data[hash]['files']:
                ext = file['path'].split('.')[-1]
                if ext not in valid_extensions:
                    continue
                lines = file['diff'].split('\n')
                for l in lines:
                    if l.startswith('+'):
                        samples.append(l[1:])

            if len(samples) == 0:
                continue

            sample = ' '.join([s for s in samples if len(s)>=minimal_token_size])
            tokens = splitter_regex.split(str(unicodedata.normalize('NFKD', sample).encode('ascii', 'ignore')))
            tokens = [t for t in tokens if t and len(t)>=minimal_token_size and len(t) <= maximal_token_size]
            tokens = [t.lower() for t in tokens if filter_regex.match(t)]

            if is_security_related:
                positive_samples.append(tokens)
                if len(positive_samples) >= batch_size:
                    with open(f'{positive_directory}/batch-{process_id}-{saved_positive_samples}.json', 'w') as f:
                        json.dump(positive_samples, f)
                    saved_positive_samples+=1
                    positive_samples=[]

            else:
                background_samples.append(tokens)
                if len(background_samples) >= batch_size:
                    with open(f'{background_directory}/batch-{process_id}-{saved_background_samples}.json', 'w') as f:
                        json.dump(background_samples, f)
                    saved_background_samples+=1
        except Exception as e:
            print('Exception')
            print(e)        
        
    with open(f'{positive_directory}/batch-{process_id}-final.json', 'w') as f:
        json.dump(positive_samples, f, indent=2)
    with open(f'{background_directory}/batch-{process_id}-final.json', 'w') as f:
        json.dump(background_samples, f, indent=2)

    return process_id


def zero_elements_under_threshhold(dictionary, threshhold):
    for k,v in dictionary.items():
        if v < threshhold:
            dictionary[k]=0


def zero_elements_with_ratio_under_threshhold(d1, d2, class_imbalance_factor,  threshhold):
    threshhold = threshhold * class_imbalance_factor
    for k,v in d1.items():
        denominator = v+d2[k]
        if denominator == 0:
            ratio = 0
        else:
            ratio = v/denominator

        if ratio < threshhold:
            d1[k]=0


def zero_elements_that_are_zero_in_collection(dictionary, checking_collection):
    for k,v in checking_collection.items():
        if v == 0:
            dictionary[k]=0


def rank_tokens():
    posi_files = get_files_in_from_directory(positive_directory)
    positive_tokens = []
    for file in posi_files:
        with open(file, 'r') as f:
            data = json.load(f)
            positive_tokens += data

    back_files = get_files_in_from_directory(background_directory)
    background_tokens = []
    for file in back_files:
        with open(file, 'r') as f:
            data = json.load(f)
            background_tokens += data
    
    all_tokens = set()
    for x in positive_tokens:
        all_tokens.update(x)
    for x in background_tokens:
        all_tokens.update(x)

    document_count_ratio = len(positive_tokens) / len(background_tokens)
    word_count_positive, tf_positive = global_term_frequency(positive_tokens, all_tokens)
    document_count_positive, df_positive = document_frequency(positive_tokens, all_tokens)
    word_count_background, tf_background = global_term_frequency(background_tokens, all_tokens)
    document_count_background, df_background = document_frequency(background_tokens, all_tokens)

    #removes rare tokens
    zero_elements_under_threshhold(document_count_positive, minimal_commit_count)
    zero_elements_under_threshhold(document_count_background, minimal_commit_count)
    zero_elements_that_are_zero_in_collection(tf_positive, document_count_positive)
    zero_elements_that_are_zero_in_collection(df_positive, document_count_positive)
    zero_elements_that_are_zero_in_collection(tf_background, document_count_background)
    zero_elements_that_are_zero_in_collection(df_background, document_count_background)
    #removes tokens common in background
    zero_elements_with_ratio_under_threshhold(document_count_positive, document_count_background, document_count_ratio, minimal_recall)

    ratio1 = dict.fromkeys(all_tokens, 0)
    ratio2 = dict.fromkeys(all_tokens, 0)
    ratio3 = dict.fromkeys(all_tokens, 0)
    zero_safe_guard = 1 / len(all_tokens)
    for x in tf_positive:
        ratio1[x] = (df_positive[x]) / (df_background[x] + zero_safe_guard)
        ratio2[x] = (tf_positive[x] * df_positive[x]) / (tf_background[x] * df_background[x] + zero_safe_guard)
        ratio3[x] = tf_positive[x] / (df_background[x] + zero_safe_guard)

    ratio_sorted = sorted(ratio2.items(), key=lambda item: -item[1])
    csv_exporatble = []
    for i in range(len(ratio_sorted)):
        kv = ratio_sorted[i]
        if kv[1] == 0:
            break

        res = {
            'token':kv[0],
            'score':kv[1],
            'positive_occurences':word_count_positive[kv[0]],
            'background_occurences':word_count_background[kv[0]],
            'positive_commits':document_count_positive[kv[0]],
            'background_commits':document_count_background[kv[0]],

        }
        csv_exporatble.append(res)

    df = pd.DataFrame(csv_exporatble)
    df.to_csv(token_ranking_file, index=False)


def document_frequency(docs, all_tokens):
    df = dict.fromkeys(all_tokens, 0)
    df_relative = dict.fromkeys(all_tokens, 0)
    docs_count = len(docs)
    
    for doc in docs:
        for word in set(doc):
            df[word] += 1
                
    for word in df:
        df_relative[word] = df[word] / docs_count

    return df, df_relative


def global_term_frequency(docs, wordset):
    docs_count = len(docs)
    tfs = dict.fromkeys(wordset, 0)
    relative_tfs = dict.fromkeys(wordset, 0)
    for doc in docs:
        for word in doc:
            tfs[word]+=1

    for word in tfs:
        relative_tfs[word] = tfs[word] / docs_count
    return tfs, relative_tfs


def mine_tokens():
    data_files = get_files_in_from_directory(data_location)
    with Pool(cpus) as p:
        with tqdm.tqdm(total=len(data_files)) as pbar:
            for _ in p.imap_unordered(pre_tokenize_samples, data_files, chunksize=1):
                pbar.update()


if __name__ == '__main__':
    start_time = time.time()
    mine_tokens()
    rank_tokens()
    print("--- %s seconds ---" % (time.time() - start_time))
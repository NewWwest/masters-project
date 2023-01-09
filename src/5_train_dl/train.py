import torch
import numpy as np
import random
import os

from src.dl.training_utils import *
from src.dl.config import *

try:
    os.mkdir(work_dir)
except FileExistsError:
    pass

try:
    os.mkdir(results_dir) 
except FileExistsError:
    pass

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def do_a_fold(i):
    global current_fold
    current_fold = i
    data_files = []
    for x in range(folds_count):
        data_files.append(load_fold_info(x))
    

    with_offset = [(x+i)%folds_count for x in range(folds_count)]
    train_data_p = []
    train_data_n = []
    for x in range(len(data_files)-2):
        train_data_p += data_files[with_offset[x]][0]
        train_data_n += data_files[with_offset[x]][1]

    train_data = [train_data_p, train_data_n]
    eval_data = data_files[with_offset[-2]]
    test_data = data_files[with_offset[-1]]

    train_dataset = load_data(train_data, oversampling_ratio, class_ratio, sample_limit)
    eval_dataset = load_data(eval_data, sample_limit=eval_sample_limit)

    file_to_embed = []
    file_to_embed += eval_data[1]
    file_to_embed += eval_data[0]
    file_to_embed += test_data[1]
    file_to_embed += test_data[0]
    
    all_files = []
    for x in data_files:
        all_files += x[0]
        all_files += x[1]


    print('FOLD', i)
    print(len(train_data[0]), len(train_data[1]))
    print(len(eval_data[0]), len(eval_data[1]))
    print(len(test_data[0]), len(test_data[1]))
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    model = fine_tune(train_dataset, eval_dataset)
    embed_files_with_model(model, all_files, f'fold-{i}')
    
    del model
    gc.collect()
    torch.cuda.empty_cache()


def do_a_fold_aggregator(i):
    data_files = []
    for x in range(folds_count):
        data_files.append(load_fold_info(x))
    
    all_files = []
    for x in data_files:
        all_files += x[0]
        all_files += x[1]


    with_offset = [(x+i)%folds_count for x in range(folds_count)]
    train_data_p = []
    train_data_n = []
    for x in range(len(data_files)-2):
        train_data_p += data_files[with_offset[x]][0]
        train_data_n += data_files[with_offset[x]][1]

    train_data = [train_data_p, train_data_n]
    eval_data = data_files[with_offset[-2]]
    test_data = data_files[with_offset[-1]]
    
    print('FOLD', 'Embedded', i)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    train_dataset, eval_dataset, test_dataset = make_commit_level_datesets(train_data, eval_data, test_data, f'fold-{i}')
    lstm_results, conv_results, mean_results = evaluate_aggregators(train_dataset, eval_dataset, test_dataset)

    
    gc.collect()
    torch.cuda.empty_cache()
    return lstm_results, conv_results, mean_results


def do_fold_finetuning():
    for i in range(folds_count):
        do_a_fold(i)


def do_folds_aggregators():    
    global current_fold
    lstm_folds_results = []
    conv_folds_results = []
    mean_folds_results = []
    for i in range(folds_count):
        current_fold = i
        lstm_results, conv_results, mean_results = do_a_fold_aggregator(i)
        lstm_folds_results.append(lstm_results)
        conv_folds_results.append(conv_results)
        mean_folds_results.append(mean_results)
        
    return lstm_folds_results, conv_folds_results, mean_folds_results



if __name__ == '__main__':
    do_fold_finetuning()
    lstm_folds_results, conv_folds_results, mean_folds_results = do_folds_aggregators()

    print('LSTM')
    print_summary(lstm_folds_results, folds_count)
    print('CONV')
    print_summary(conv_folds_results, folds_count)
    print('MEAN')
    print_summary(mean_folds_results, folds_count)

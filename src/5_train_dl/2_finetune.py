#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.utils.data as data
import torch
import time
import tqdm
import pandas as pd
import numpy as np
import json
import random

from src.dl.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.dl.datasets.supporting.CsvDataset import CsvDataset
from src.dl.datasets.sampling.OverSampledDataset import OverSampledDataset
from src.dl.datasets.sampling.UnderSampledDataset import UnderSampledDataset

from src.dl.datasets.load import load_sample_level
from src.dl.dl_utils import save_dataset, read_dataset
from src.dl.models.BertAndLinear import BertAndLinear as WorkingModel
from src.utils.utils import get_files_in_from_directory

# Model config
base_model = 'microsoft/graphcodebert-base'
batch_size_ = 4
num_epochs_ = 3
fraction_of_data = 0.05
train_percentage_size = 0.9
class_ratio = 1
learning_rate = 2e-5

# Output&Evaluation configuration
save_model_in_each_epoch = True
eval_model_in_each_epoch = True
model_name = 't3_roll_java'
work_dir = f'src/rq5/binaries/{model_name}'

# Data config - Set to None if you want to use cached datasets
# raw_input_path = 'results/dl/CodeParserMiner_ast'
# raw_input_path = 'results/dl/CodeParserMiner_edit'
# raw_input_path = 'results/dl/AddedCodeMiner'
raw_input_path = 'results/dl/RollingWindowMiner'
# raw_input_path = None


# jsut for 'just_eval':
model_to_eval = 'epoch_1'
# eval_input_path = 'results/dl/AddedCodeMiner_test'
eval_input_path = None


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_model(model, optimizer, data_loader, loss_module, scheduler, eval_loader = None):
    torch.cuda.empty_cache()
    model.train()
    model.to(device)

    accumulated_loss = 0
    all_samples = 0
    positive_samples = 0

    for epoch in range(num_epochs_):
        print(f'Epoch {epoch}/{num_epochs_}')
        accumulated_loss = 0

        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for data_inputs, data_labels in data_loader:
                # Step 0: Diagnostics :x
                positive_samples += len([1 for x in data_labels if x[0] == 1])
                all_samples += len(data_labels)
                
                # Step 1: Mode data to device
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                # Step 2: Calculate model output
                preds = model(data_inputs)
                preds = preds.squeeze(dim=0)

                # Step 3: Calculate loss
                loss = loss_module(preds, data_labels.float())
                accumulated_loss += loss.item()

                ## Step 4: Perform backpropagation
                optimizer.zero_grad()
                loss.backward()

                ## Step 5: Update the parameters
                optimizer.step()
                scheduler.step()
                
                # Step 6: Progress bar
                pbar.update()
        print('Loss in this epoch:', accumulated_loss)

        if save_model_in_each_epoch:
            torch.save(model.state_dict(), f'{work_dir}/model_{model_name}_epoch_{epoch}.pickle')

        if eval_loader != None:
            eval_model(model, eval_loader)


    print(f'Model saw positive samples {positive_samples} times and background samples {all_samples-positive_samples}')
    print(f'Ratio 1:{(all_samples-positive_samples)/positive_samples}')


def eval_model(model, data_loader):
    torch.cuda.empty_cache()
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []
    data_size = len(data_loader)
    with tqdm.tqdm(total=data_size) as pbar:
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=0)

            labels_in_memory = data_labels.cpu().detach().numpy()
            if len(labels_in_memory.shape) == 1:
                all_labels.append(labels_in_memory)
            else:
                for x in labels_in_memory:
                    all_labels.append(x)
                    
            preds_in_memory = preds.cpu().detach().numpy()
            if labels_in_memory.shape[0] == 1:
                all_predictions.append(preds_in_memory)
            else:
                for x in preds_in_memory:
                    all_predictions.append(x)

            pbar.update()

    predictions_arr = [1 if x[0]>x[1] else 0 for x in all_predictions]
    targets_arr = [1 if x[0]>x[1] else 0 for x in all_labels]
    P = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==1])
    TP = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==1 and targets_arr[x]==1])
    FP = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==1 and targets_arr[x]==0])
    FN = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==0 and targets_arr[x]==1])
    TN = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==0 and targets_arr[x]==0])
    N = len([1 for x in range(len(predictions_arr)) if predictions_arr[x]==0])

    precission = TP/(TP+FP) if (TP+FP)!=0 else 0
    recall = TP/(TP+FN) if (TP+FN)!=0 else 0
    print('Precission:',f'{TP}/{TP+FP}', precission)
    print('Recall', f'{TP}/{TP+FN}', recall)
    print(f'P:{P},', f'TP:{TP},', f'FP:{FP},', f'FN:{FN},', f'TN:{TN},', f'N:{N}')

    return precission, recall


def load_data(input_path):
    # Load Data
    dataset, test_dataset = load_sample_level(input_path)

    # Data limit for testing
    # if fraction_of_data < 1:
    #     dataset, rejected_data = dataset.split_data(fraction_of_data)
    dataset.crop_data(5000, 6000)
    print('dataset.positive_data', len(dataset.positive_data))
    print('dataset.background_data', len(dataset.background_data))

    # Train/Test split & rebalancing
    train_dataset, eval_dataset = dataset.split_data(train_percentage_size)
    train_dataset = UnderSampledDataset(train_dataset, class_ratio)

    # Save the data
    save_dataset(train_dataset, f'{work_dir}/train_dataset_{model_name}.csv')
    save_dataset(eval_dataset, f'{work_dir}/eval_dataset_{model_name}.csv')
    save_dataset(test_dataset, f'{work_dir}/test_dataset_{model_name}.csv')

    return train_dataset, eval_dataset, test_dataset


def finetune_and_eval():
    # Load data
    if raw_input_path != None:
        train_dataset, eval_dataset , test_dataset = load_data(raw_input_path)
    else:
        train_dataset = read_dataset(f'{work_dir}/train_dataset_{model_name}.csv')
        eval_dataset = read_dataset(f'{work_dir}/eval_dataset_{model_name}.csv')
        test_dataset = read_dataset(f'{work_dir}/test_dataset_{model_name}.csv')
    
    # Define model
    model = WorkingModel(base_model)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
        num_warmup_steps=int(len(train_dataset)*0.25), 
        num_training_steps=len(train_dataset)*num_epochs_)

    # Prep the loaders
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size_, drop_last=True, shuffle=True)
    if eval_model_in_each_epoch:
        eval_data_loader = data.DataLoader(eval_dataset, batch_size=batch_size_, drop_last=True, shuffle=True)
    else:
        eval_data_loader = None

    # Train the model
    train_model(model, optimizer, train_data_loader, loss_module, scheduler, eval_loader=eval_data_loader)
    torch.save(model.state_dict(), f'{work_dir}/model_{model_name}_final.pickle')

    # Test the model on eval subset
    eval_data_loader = data.DataLoader(eval_dataset, drop_last=True, batch_size=batch_size_)
    eval_model(model, eval_data_loader)

    # Test the model on test subset
    test_data_loader = data.DataLoader(test_dataset, drop_last=True, batch_size=batch_size_)
    eval_model(model, test_data_loader)


def just_eval():
    model = WorkingModel(base_model)
    model.load_state_dict(torch.load(f'{work_dir}/model_{model_name}_{model_to_eval}.pickle'))

    positive_json_files = get_files_in_from_directory(eval_input_path, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(eval_input_path, extension='.json', startswith='background-encodings')
    eval_dataset = SampleLevelRawDataset()
    eval_dataset.load_files(positive_json_files, background_json_files)

    eval_data_loader = data.DataLoader(eval_dataset, drop_last=True, batch_size=batch_size_)
    eval_model(model, eval_data_loader)



if __name__ == '__main__':
    start_time = time.time()
    finetune_and_eval()
    # just_eval()
    print("--- %s seconds ---" % (time.time() - start_time))



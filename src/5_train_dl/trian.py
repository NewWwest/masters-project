import sys
sys.path.insert(0, r'D:\Projects\aaa')

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
import os

from src.dl.datasets.CommitLevelRawDataset import CommitLevelRawDataset
from src.dl.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.dl.datasets.supporting.CsvDataset import CsvDataset
from src.dl.datasets.sampling.OverSampledDataset import OverSampledDataset
from src.dl.datasets.sampling.UnderSampledDataset import UnderSampledDataset

from src.dl.dl_utils import save_dataset, read_dataset, get_repo_seminames, get_files_in_set
from src.dl.models.BertAndLinear import BertAndLinear as FineTuningModel
from src.dl.models.LstmAggregator import LstmAggregator as AggregatorModel
from src.utils.utils import get_files_in_from_directory

# Model config
base_model = 'microsoft/graphcodebert-base'
batch_size_ = 4
num_epochs_ = 3
fraction_of_data = 1

test_percentage = 0.15
eval_percentage = 0.05


class_ratio = 2
learning_rate = 2e-6


aggregator_num_epochs_ = 15
aggregator_class_ratio = 2
aggregator_learning_rate = 2e-5

save_model_in_each_epoch = True
eval_model_in_each_epoch = True
model_name = 'testsq'
work_dir = f'src/rq5/binaries/{model_name}'

results_dir = f'src/rq5/binaries/testEmbeddings'

# Data config - Set to None if you want to use cached datasets
raw_input_path = 'results/dl/CodeParserMiner_ast'
# raw_input_path = 'results/dl/CodeParserMiner_edit'
# raw_input_path = 'results/dl/AddedCodeMiner'
# raw_input_path = 'results/dl/RollingWindowMiner'
# raw_input_path = None


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


def load_files(input_path, data_fraction=1):
    positive_json_files = get_files_in_from_directory(input_path, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(input_path, extension='.json', startswith='background-encodings')

    if data_fraction < 1:
        positive_json_files = random.sample(positive_json_files, int(len(positive_json_files)*data_fraction))
        background_json_files = random.sample(background_json_files, int(len(background_json_files)*data_fraction))


    repos_set = get_repo_seminames(positive_json_files)
    repos_count = len(repos_set)


    repos_test = set(random.sample(list(repos_set), int(repos_count*test_percentage)))
    repos_set.difference_update(repos_test)
    repos_eval = set(random.sample(list(repos_set), int(repos_count*test_percentage)))
    repos_set.difference_update(repos_eval)

    positive_train = get_files_in_set(positive_json_files, repos_set)
    positive_eval = get_files_in_set(positive_json_files, repos_eval)
    positive_test = get_files_in_set(positive_json_files, repos_test)

    background_train = get_files_in_set(background_json_files, repos_set)
    background_eval = get_files_in_set(background_json_files, repos_eval)
    background_test = get_files_in_set(background_json_files, repos_test)

    return (positive_json_files, background_json_files), (positive_train, background_train), (positive_eval, background_eval), (positive_test, background_test)


def load_data(input_data, positive_samples_limit=None, background_samples_limit=None):
    positive_files = input_data[0]
    background_files = input_data[1]

    dataset = SampleLevelRawDataset()
    dataset.load_files(positive_files, background_files)

    if background_samples_limit != None and positive_samples_limit != None:
        dataset.crop_data(positive_samples_limit, background_samples_limit)   

    return dataset


def embed_files(tokenizer, data_files):
    with tqdm.tqdm(total=len(data_files)) as pbar:
        for data_file in data_files:
            with open(data_file, 'r') as f:
                data = json.load(f)

            embeddings = []
            for data_point in data:
                if 'commit_sample' in data_point and \
                    data_point['commit_sample'] != None and \
                    len(data_point['commit_sample']) > 0:

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

            pbar.update()


def map_files_to_new_repo(data_files):
    new_data_files = []
    for data_file in data_files:
        file_name = os.path.basename(data_file)
        new_file = os.path.join(results_dir, 'embedded-' + file_name)
        if os.path.exists(new_file):
            new_data_files.append(new_file)

    return new_data_files


def train_model2(model, optimizer, data_loader, loss_module, scheduler, test_loader = None):
    torch.cuda.empty_cache()
    model.train()
    model.to(device)

    accumulated_loss = 0
    all_samples = 0
    positive_samples = 0

    for epoch in range(aggregator_num_epochs_):
        print(f'Epoch {epoch}/{aggregator_num_epochs_}')
        accumulated_loss = 0
        model.train()

        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for data_inputs, data_labels in data_loader:
                # Step 0: Diagnostics :x
                positive_samples += len([1 for x in data_labels if x[0] == 1])
                all_samples += len(data_labels)

                #TODO different commit mode and sample mode
                data_inputs = torch.stack(data_inputs)
                
                # Step 1: Mode data to device 
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device) 

                # Step 2: Calculate model output
                preds = model(data_inputs)
                
                #TODO different commit mode and sample mode
                # preds = preds.squeeze(dim=0)

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

        if test_loader != None:
            eval_model2(model, test_loader)


    print(f'Model saw positive samples {positive_samples} times and background samples {all_samples-positive_samples}')
    print(f'Ratio 1:{(all_samples-positive_samples)/positive_samples}')


def eval_model2(model, data_loader):
    torch.cuda.empty_cache()
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []
    data_size = len(data_loader)
    with tqdm.tqdm(total=data_size) as pbar:
        for data_inputs, data_labels in data_loader:

            #TODO different commit mode and sample mode
            data_inputs = torch.stack(data_inputs)

            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            #TODO different commit mode and sample mode
            # preds = preds.squeeze(dim=0)

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

    #TODO different commit mode and sample mode
    predictions_arr = [1 if x[0,0]>x[0,1] else 0 for x in all_predictions]
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



def do_stuff():
    # Load data
    all_data, train_data, eval_data, test_data = load_files(raw_input_path, fraction_of_data)

    train_dataset = load_data(train_data)
    train_dataset = UnderSampledDataset(train_dataset, class_ratio)

    eval_dataset = load_data(eval_data)
    test_dataset = load_data(test_data)

    # Define model
    model = FineTuningModel(base_model)
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
        num_warmup_steps=int(len(train_dataset)*0.25), 
        num_training_steps=len(train_dataset)*num_epochs_)

    # Prep the loaders
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size_, drop_last=True, shuffle=True)
    eval_data_loader = data.DataLoader(eval_dataset, batch_size=batch_size_, drop_last=True, shuffle=True)

    # Train the model
    train_model(model, optimizer, train_data_loader, loss_module, scheduler, eval_loader=eval_data_loader)
    torch.save(model.state_dict(), f'{work_dir}/model_{model_name}_final.pickle')

    # Test the model on test subset
    test_data_loader = data.DataLoader(test_dataset, drop_last=True, batch_size=batch_size_)
    eval_model(model, test_data_loader)

    tokenizer = model.codebert
    for param in tokenizer.parameters():
        param.requires_grad = False
    tokenizer.eval()
    tokenizer.to(device)
    
    print('Embedding test set')
    embed_files(tokenizer, test_data[0])
    embed_files(tokenizer, test_data[1])
    print('Embedding evaluation set')
    embed_files(tokenizer, eval_data[0])
    embed_files(tokenizer, eval_data[1])
    print('Embedding train set')
    embed_files(tokenizer, train_data[0])
    embed_files(tokenizer, train_data[1])

    # convert sample files to embedded files
    train_data, eval_data, test_data
    train_data_embeded_pos = map_files_to_new_repo(train_data[0])
    train_data_embeded_bac = map_files_to_new_repo(train_data[1])

    eval_data_embeded_pos = map_files_to_new_repo(eval_data[0])
    eval_data_embeded_bac = map_files_to_new_repo(eval_data[1])

    test_data_embeded_pos = map_files_to_new_repo(test_data[0])
    test_data_embeded_bac = map_files_to_new_repo(test_data[1])


    train_dataset_embeded = CommitLevelRawDataset()
    train_dataset_embeded.load_files(train_data_embeded_pos, train_data_embeded_bac)
    train_dataset_embeded = UnderSampledDataset(train_dataset_embeded, aggregator_class_ratio)
    eval_dataset_embeded = CommitLevelRawDataset()
    eval_dataset_embeded.load_files(eval_data_embeded_pos, eval_data_embeded_bac)
    test_dataset_embeded = CommitLevelRawDataset()
    test_dataset_embeded.load_files(test_data_embeded_pos, test_data_embeded_bac)


    
    # Define model
    model = AggregatorModel()
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=aggregator_learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
        num_warmup_steps=int(len(train_dataset)*0.25), 
        num_training_steps=len(train_dataset)*num_epochs_)

    # Prep the loaders
    train_data_embeded_loader = data.DataLoader(train_dataset_embeded, batch_size=1, drop_last=True, shuffle=True)
    eval_data_embeded_loader = data.DataLoader(eval_dataset_embeded, batch_size=1, drop_last=True, shuffle=True)

    train_model2(model, optimizer, train_data_embeded_loader, loss_module, scheduler, test_loader=eval_data_embeded_loader)
    torch.save(model.state_dict(), f'{work_dir}/model_aggregator_{model_name}_final.pickle')

    # Test the model on eval subset
    test_data_embeded_loader = data.DataLoader(test_dataset_embeded, drop_last=True, batch_size=1)
    eval_model2(model, test_data_embeded_loader)






if __name__ == '__main__':
    start_time = time.time()
    do_stuff()
    print("--- %s seconds ---" % (time.time() - start_time))



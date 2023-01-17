# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
from transformers import get_linear_schedule_with_warmup
import gc
import torch.nn as nn
import torch.utils.data as data
import torch
import tqdm
import json
import random
import os


from src.dl.datasets.BaseRawDataset import BaseRawDataset
from src.dl.datasets.SampleLevelRawDataset import SampleLevelRawDataset
from src.dl.datasets.CommitLevelRawDataset import CommitLevelRawDataset
from src.dl.datasets.sampling.OverSampledDataset import OverSampledDataset
from src.dl.datasets.sampling.UnderSampledDataset import UnderSampledDataset

from src.dl.models.BertAndLinear import BertAndLinear
from src.dl.models.LstmAggregator import LstmAggregator
from src.dl.models.ConvAggregator import ConvAggregator
from src.dl.models.MeanAggregator import MeanAggregator

from src.dl.config import *


def train_model(model, optimizer, data_loader, loss_module, scheduler, eval_loader = None):
    '''
    Function for fine tuning the BERT models. 
    If eval_loader is provide the model will be evaluated after each epoch.
    '''
    torch.cuda.empty_cache()
    global current_fold

    model.train()
    model.to(device)

    accumulated_loss = 0
    all_samples = 0
    positive_samples = 0

    for epoch in range(num_epochs_):
        print(f'Epoch {epoch}/{num_epochs_}')
        accumulated_loss = 0

        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for data_inputs_x, data_labels in data_loader:
                commit_ids = data_inputs_x[0]
                data_inputs = data_inputs_x[1]

                # Step 0: Diagnostics
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
            torch.save(model.state_dict(), f'{work_dir}/model_{model_name}_fold_{current_fold}_epoch_{epoch}.pickle')

        if eval_loader != None:
            eval_model(model, eval_loader)


    print(f'Model saw positive samples {positive_samples} times and background samples {all_samples-positive_samples}')
    print(f'Ratio 1:{(all_samples-positive_samples)/positive_samples}')


def eval_model(model, data_loader):
    '''
    Function for evaluting the BERT models. 
    It detemines the class assigned by the model by comparing the output of the model and taking the class for which the value is higher.
    '''
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []
    data_size = len(data_loader)
    with tqdm.tqdm(total=data_size) as pbar:
        for data_inputs_x, data_labels in data_loader:
            commits_ids = data_inputs_x[0]
            data_inputs = data_inputs_x[1]

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


def load_data(input_data, oversampling_ratio=None, class_ratio=None, sample_limit=None):
    '''
    Loads the data from the files provided in the function.
    It expects a pair where elements are a collection of files to load into memory.
    The first element are the positive samples, second are the background.
    Data re-balancing can be done by providing specific ratios.
    '''
    positive_files = input_data[0]
    background_files = input_data[1]

    dataset = SampleLevelRawDataset()
    dataset.load_files(positive_files, background_files)

    if oversampling_ratio != None and class_ratio != None and sample_limit != None:
        dataset.setup_ratios(oversampling_ratio, class_ratio, sample_limit)   

    if oversampling_ratio == None and class_ratio == None and sample_limit != None:
        dataset.limit_data(sample_limit)   

    return dataset


def embed_files(tokenizer, data_files, marker):
    '''
    Processes the collection files with the provided tokenizer(embedder).
    The marker is a string that will be used as the start of the file name, used in other fuctions to denote the epoch.
    '''
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
                new_file = os.path.join(results_dir, marker + '-embedded-' + file_name)
                with open(new_file, 'w') as f:
                    json.dump(embeddings, f)

            pbar.update()


def embed_files_with_model(model, files, marker):
    '''
    Calls embed_files, but before sets the model into evaluation mode.
    '''
    # Test the model on test subset
    for param in model.codebert.parameters():
        param.requires_grad = False
    model.codebert.eval()
    model.codebert.to(device)

    print('Embedding files with transformer from', marker)
    embed_files(model.codebert, files, marker)


def map_files_to_embedded_input(data_files, marker):
    '''
    Converts the input filenames into the filenames of embedded files.
    '''
    new_data_files = []
    for data_file in data_files:
        file_name = os.path.basename(data_file)
        new_file = os.path.join(results_dir, marker + '-embedded-' + file_name)
        if os.path.exists(new_file):
            new_data_files.append(new_file)

    return new_data_files


def load_fold_info(fold_number):
    '''
    Loads the lists of the filenames used in each fold.
    If the configuration states not to use all the data, some files are randomly omitted.
    '''
    with open(os.path.join(raw_input_path, f'fold-{fold_number}-files.json'), 'r') as f:
        data = json.load(f)

    all_positives = data['positive_files']
    all_backgrounds = data['background_files']
    # all_positives = [os.path.join(repo_dir, x) for x in all_positives]
    # all_backgrounds = [os.path.join(repo_dir, x) for x in all_backgrounds]
    
    if fraction_of_data < 1:
        positives = random.sample(all_positives, int(fraction_of_data*len(all_positives)))
        backgrounds = random.sample(all_backgrounds, int(fraction_of_data*len(all_backgrounds)))
    else:
        positives = all_positives
        backgrounds = all_backgrounds

    return (positives, backgrounds)
    

def fine_tune(train_dataset, eval_dataset):
    '''
    Setup function for fine tuning, where the models, loss and loaders are created.
    '''
    global current_fold

    # Define model
    model = BertAndLinear(base_model)
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
    torch.save(model.state_dict(), f'{work_dir}/model_{model_name}_fold_{current_fold}_final.pickle')
    return model


def make_commit_level_datesets(train_data, eval_data, test_data, marker):
    '''
    Converts the list of files into datasets
    '''
    train_data_embeded_pos = map_files_to_embedded_input(train_data[0], marker)
    train_data_embeded_bac = map_files_to_embedded_input(train_data[1], marker)

    eval_data_embeded_pos = map_files_to_embedded_input(eval_data[0], marker)
    eval_data_embeded_bac = map_files_to_embedded_input(eval_data[1], marker)

    test_data_embeded_pos = map_files_to_embedded_input(test_data[0], marker)
    test_data_embeded_bac = map_files_to_embedded_input(test_data[1], marker)


    train_dataset_embeded = CommitLevelRawDataset()
    train_dataset_embeded.load_files(train_data_embeded_pos, train_data_embeded_bac)
    train_dataset_embeded = UnderSampledDataset(train_dataset_embeded, aggregator_class_ratio)
    eval_dataset_embeded = CommitLevelRawDataset()
    eval_dataset_embeded.load_files(eval_data_embeded_pos, eval_data_embeded_bac)
    test_dataset_embeded = CommitLevelRawDataset()
    test_dataset_embeded.load_files(test_data_embeded_pos, test_data_embeded_bac)
    return train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded


def train_aggregator(model, optimizer, data_loader, loss_module, scheduler, test_loader = None):
    '''
    Function for training the aggregator models. 
    If test_loader is provide the model will be evaluated after each epoch.
    The function is similar to the sample level finetuningm however some changes are present as in this case each input is a collection of samples.
    '''
    global current_fold
    torch.cuda.empty_cache()

    model.train()
    model.to(device)

    accumulated_loss = 0
    all_samples = 0
    positive_samples = 0
    results = []

    for epoch in range(aggregator_num_epochs_):
        print(f'Epoch {epoch}/{aggregator_num_epochs_}')
        accumulated_loss = 0
        model.train()

        with tqdm.tqdm(total=len(data_loader)) as pbar:
            for data_inputs_x, data_labels in data_loader:
                commits_id = data_inputs_x[0]
                data_inputs = data_inputs_x[1]

                # Step 0: Diagnostics
                positive_samples += len([1 for x in data_labels if x[0] == 1])
                all_samples += len(data_labels)

                # In aggregated stage the input has to be stacked into one vector 
                data_inputs = torch.stack(data_inputs)
                
                # Step 1: Mode data to device 
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device) 

                # Step 2: Calculate model output
                # In aggregated stage the prediction is already in the correct dimensions
                preds = model(data_inputs)
                
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
            torch.save(model.state_dict(), f'{work_dir}/model_agg_{model_name}_fold_{current_fold}_epoch_{epoch}.pickle')

        eval_set_loss, precission, recall = eval_aggregator(model, test_loader, loss_module)
        results.append({
            'epoch':epoch,
            'eval_set_loss':eval_set_loss,
            'precission':precission,
            'recall':recall,
        })


    print(f'Model saw positive samples {positive_samples} times and background samples {all_samples-positive_samples}')
    print(f'Ratio 1:{(all_samples-positive_samples)/positive_samples}')
    return results


def eval_aggregator(model, data_loader, loss_module):
    '''
    Function for evaluating the aggregator models. 
    It detemines the class assigned by the model by comparing the output of the model and taking the class for which the value is higher.
    The function is similar to the sample level finetuningm however some changes are present as in this case each input is a collection of samples.
    '''
    torch.cuda.empty_cache()
    model.eval()
    model.to(device)

    all_labels = []
    all_predictions = []
    data_size = len(data_loader)
    accumulated_loss = 0
    with tqdm.tqdm(total=data_size) as pbar:
        for data_inputs_x, data_labels in data_loader:
            commits_id = data_inputs_x[0]
            data_inputs = data_inputs_x[1]

            data_inputs = torch.stack(data_inputs)

            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            preds = model(data_inputs)
            
            loss = loss_module(preds, data_labels.float())
            lossxd = float(loss.item())
            accumulated_loss += lossxd

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
    print('Loss:', accumulated_loss)
    print('Precission:',f'{TP}/{TP+FP}', precission)
    print('Recall', f'{TP}/{TP+FN}', recall)
    print(f'P:{P},', f'TP:{TP},', f'FP:{FP},', f'FN:{FN},', f'TN:{TN},', f'N:{N}')

    return accumulated_loss, precission, recall


def train_and_eval_aggregator(model, train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded):
    '''
    Wrapped function for training and evaluating the aggregator models. 
    After the training is done the epoch in which the loss on the eval dataset was lowest is chosen. The model from this epoch is then evaluated on the test dataset.
    '''
    global current_fold
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=aggregator_learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
        num_warmup_steps=int(len(train_dataset_embeded)*0.25), 
        num_training_steps=len(train_dataset_embeded)*num_epochs_)

    # Prep the loaders
    train_data_embeded_loader = data.DataLoader(train_dataset_embeded, batch_size=1, drop_last=True, shuffle=True)
    eval_data_embeded_loader = data.DataLoader(eval_dataset_embeded, batch_size=1, drop_last=True, shuffle=True)

    performance_results = train_aggregator(model, optimizer, train_data_embeded_loader, loss_module, scheduler, test_loader=eval_data_embeded_loader)

    lowest_loss = 1_000_000
    lowest_loss_epoch = 0
    for res in performance_results:
        if lowest_loss > res['eval_set_loss']:
            lowest_loss = res['eval_set_loss']
            lowest_loss_epoch = res['epoch']


    model_path = f'{work_dir}/model_agg_{model_name}_fold_{current_fold}_epoch_{lowest_loss_epoch}.pickle'
    model.load_state_dict(torch.load(model_path))
    test_data_embeded_loader = data.DataLoader(test_dataset_embeded, drop_last=True, batch_size=1)
    accumulated_loss, precission, recall = eval_aggregator(model, test_data_embeded_loader, loss_module)

    return accumulated_loss, precission, recall


def evaluate_aggregators(train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded):
    '''
    Calls train_and_eval_aggregator on each aggregator model. 
    It's possible to comment out the models that one does not want to be trained in this run
    '''
    lstm_results = []
    model = LstmAggregator()
    lstm_results = train_and_eval_aggregator(model, train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded)
    del model

    conv_results = []
    model = ConvAggregator()
    conv_results = train_and_eval_aggregator(model, train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded)
    del model

    mean_results = []
    model = MeanAggregator()
    mean_results = train_and_eval_aggregator(model, train_dataset_embeded, eval_dataset_embeded, test_dataset_embeded)
    del model

    return (lstm_results, conv_results, mean_results)


def f1_score(precission, recall):
    return 0 if precission+recall==0  else 2*precission*recall/(precission+recall)


def print_summary(resutls, folds):
    '''Prints the results of aggregator training'''
    print('avg_accumulated_loss', sum([x[0] for x in resutls])/folds)
    avg_precission = sum([x[1] for x in resutls])/folds
    avg_recall = sum([x[2] for x in resutls])/folds
    print('avg_precission', avg_precission)
    print('avg_recall', avg_recall)
    f1_scores = sum([f1_score(x[1], x[2]) for x in resutls])/folds
    print('avg_f1', f1_scores)
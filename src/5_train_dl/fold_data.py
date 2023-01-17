#!/usr/bin/env python3
#
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

# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import time
import numpy as np
import os
import random

from src.dl.dl_utils import get_repo_seminames, chunks, get_files_in_set, save_file_datasets
from src.utils.utils import get_files_in_from_directory

fraction_of_data = 0.2
folds_count = 5

model_name = 'debug_test_folds'
work_dir = f'src/5_train_dl/binaries/{model_name}'
raw_input_path = r'results\\run_1'


random.seed(42)
np.random.seed(42)

try: 
    os.makedirs(work_dir)
except: 
    pass


def load_fold_data(input_path, fold_count = 5,  data_fraction=1):
    positive_json_files = get_files_in_from_directory(input_path, extension='.json', startswith='positive-encodings')
    background_json_files = get_files_in_from_directory(input_path, extension='.json', startswith='background-encodings')

    if data_fraction < 1:
        positive_json_files = random.sample(positive_json_files, int(len(positive_json_files)*data_fraction))
        background_json_files = random.sample(background_json_files, int(len(background_json_files)*data_fraction))


    repos_set = get_repo_seminames(positive_json_files)
    repos_list = list(repos_set)
    random.shuffle(repos_list)

    result = []
    repos_folded = chunks(repos_list, fold_count)
    for fold in repos_folded:
        fold_set = set(fold)
        fold_positive = get_files_in_set(positive_json_files, fold_set)
        fold_background = get_files_in_set(background_json_files, fold_set)
        result.append((fold_positive, fold_background))

    return result
   

def main():
    data_files = load_fold_data(raw_input_path, fold_count=folds_count, data_fraction=fraction_of_data)

    for i in range(folds_count):
        save_file_datasets(data_files[i], f'fold-{i}', work_dir)
        

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

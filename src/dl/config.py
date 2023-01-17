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
import sys
import torch

repo_dir = f'D:\\Projects\\masters'

sys.path.insert(0, repo_dir)

# Model config
base_model = 'microsoft/graphcodebert-base'
batch_size_ = 2
num_epochs_ = 3

fraction_of_data = 1

sample_limit = 1_000_000
eval_sample_limit = 1_000_000
folds_count = 5
current_fold = -1

learning_rate = 2e-6
oversampling_ratio = None # if None, no ratio controll will be applied
class_ratio = 5

aggregator_num_epochs_ = 5
aggregator_class_ratio = 5
aggregator_learning_rate = 2e-4

save_model_in_each_epoch = True
eval_model_in_each_epoch = True

model_guid = 'debug_run'
model_name = model_guid

work_dir = f'{repo_dir}\\src\\5_train_dl\\binaries\\{model_name}'
results_dir = f'{repo_dir}\\src\\5_train_dl\\binaries\\data{model_name}'
raw_input_path = f'{repo_dir}\\src\\5_train_dl\\binaries\\debug_test_folds'

seed = 42
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
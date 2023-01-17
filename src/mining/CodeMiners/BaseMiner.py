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
from abc import abstractmethod
from pydriller import RepositoryMining, Commit
import pandas as pd
import json

class BaseMiner:
    def __init__(self, sample_encodder):
        self.sample_encodder = sample_encodder


    @abstractmethod
    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        pass

    def save(self, checkpoints_directory, owner, repo, commit, label_security_related, commit_data):
        if commit_data == None or len(commit_data) == 0:
            return

        flag = 'positive' if label_security_related else 'background'
        sha = commit.hash
        path_to_new_file = f'{checkpoints_directory}/{flag}-samples-{owner}-{repo}-{sha}.json'
        with open(path_to_new_file, 'w') as f:
            json.dump(commit_data, f)

        return [path_to_new_file]
            

    def save_and_tokenize(self, checkpoints_directory, owner, repo, commit, label_security_related, commit_data):
        if commit_data == None or len(commit_data) == 0:
            return

        datafiles1 = self.save(checkpoints_directory, owner, repo, commit, label_security_related, commit_data)
            
        flag = 'positive' if label_security_related else 'background'
        sha = commit.hash
        commit_data_encodded = []
        for sample in commit_data:
            tokens = self.sample_encodder.process_sample(sample)
            res1 = {
                    'commit_id': sample['commit_id'],
                    'sample_type': sample['sample_type'],
                    'file_name': sample['file_name'],
                    'is_security_related': sample['is_security_related'],
                    'commit_sample': tokens
                }
            commit_data_encodded.append(res1)

        path_to_new_file =f'{checkpoints_directory}/{flag}-encodings-{owner}-{repo}-{sha}.json'
        with open(path_to_new_file, 'w') as f:
            json.dump(commit_data_encodded, f)

        return [path_to_new_file] + datafiles1


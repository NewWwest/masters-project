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
from pydriller import Commit
import random

from src.mining.CodeMiners.BaseMiner import BaseMiner

class VulFixMinerMiner(BaseMiner):
    def __init__(self, results_dir, valid_extensions, extensions_to_ignore):
        super().__init__(None)
        self.checkpoints_directory = results_dir
        self.valid_extensions = set(valid_extensions) if valid_extensions else None
        self.extensions_to_ignore = set(extensions_to_ignore) if extensions_to_ignore else None


    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        commit_id = f'{owner}/{repo}/{commit.hash}'
        commit_message_cropped = commit.msg[0:min(256,len(commit.msg))]

        changeFiles = []
        for changeFile in random.sample(commit.modifications, min(len(commit.modifications), 20)):
            if self.valid_extensions != None and changeFile.filename.split('.')[-1] not in self.valid_extensions:
                continue

            if self.extensions_to_ignore != None and changeFile.filename.split('.')[-1] in self.extensions_to_ignore:
                continue

            res1 = {
                'id': commit_id,
                'sample_type': 'VulFixMinerMiner',
                'file_name': changeFile.filename,
                'is_security_related': label_security_related,
                'commit_message': commit_message_cropped,
                'commit_patch': changeFile.diff
            }
            changeFiles.append(res1)
        
        return self.save(self.checkpoints_directory, owner, repo, commit, label_security_related, changeFiles)


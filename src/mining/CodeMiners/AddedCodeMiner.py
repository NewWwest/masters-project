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

from src.mining.CodeMiners.BaseMiner import BaseMiner

class AddedCodeMiner(BaseMiner):
    def __init__(self, results_dir, sample_encodder, valid_extensions, extensions_to_ignore):
        super().__init__(sample_encodder)
        self.checkpoints_directory = results_dir
        self.valid_extensions = set(valid_extensions) if valid_extensions else None
        self.extensions_to_ignore = set(extensions_to_ignore) if extensions_to_ignore else None


    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        commit_id = f'{owner}/{repo}/{commit.hash}'
        commit_first_line = commit.msg.split('\n')[0]
        commit_title = commit_first_line[0:min(72,len(commit_first_line))]

        changeFiles = []
        for changeFile in commit.modifications:
            safe_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path

            if self.valid_extensions != None and changeFile.filename.split('.')[-1] not in self.valid_extensions:
                continue

            if self.extensions_to_ignore != None and changeFile.filename.split('.')[-1] in self.extensions_to_ignore:
                continue

            if 'added' not in changeFile.diff_parsed:
                continue

            lines = [x[1] for x in changeFile.diff_parsed['added']]
            res1 = {
                'commit_id': commit_id,
                'sample_type': 'AddedCodeMiner',
                'file_name': safe_path,
                'is_security_related': label_security_related,
                'commit_title': commit_title,
                'commit_sample': '\n'.join(lines)
            }
            changeFiles.append(res1)
        
        return self.save_and_tokenize(self.checkpoints_directory, owner, repo, commit, label_security_related, changeFiles)


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
import pandas as pd 

class SecurityRelatedCommitsLoader:
    def __init__(self, mapping_file) -> None:
        mapping_csv = pd.read_csv(mapping_file)

        self._commits = {}
        for i, r in mapping_csv.iterrows():
            repo_full_name = f'{r["repo_owner"]}/{r["repo_name"]}'
            if repo_full_name not in self._commits:
                self._commits[repo_full_name] = []
            
            self._commits[repo_full_name].append(r)


    def get_repos_with_atleastn_fix_commits(self, n):
        with_fix_count = []
        for x in self._commits:
            if len(self._commits[x]) >= n:
                with_fix_count.append((x, len(self._commits[x])))

        with_fix_count = sorted(with_fix_count, key=lambda x: x[1], reverse=True)
        return [x[0] for x in with_fix_count]


    def shas_for_repo(self, repo_full_name):
        if repo_full_name not in self._commits:
            return []

        return list([x['commit_sha'] for x in self._commits[repo_full_name]])

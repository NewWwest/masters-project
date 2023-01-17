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

import json
import pandas as pd
import time
from src.utils.utils import get_files_in_from_directory
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
from src.proxies.GitHubProxy import GithubProxy


input_issues_references_dataset = [
    r'data\most_starred\manually_annotated_mapping.csv',
    r'data\most_used_npm\manually_annotated_mapping.csv',
    r'data\most_used_pypi\manually_annotated_mapping.csv',
    r'data\most_used_mvn\manually_annotated_mapping.csv',
]

def main():
    issues_df = pd.concat([pd.read_csv(x) for x in input_issues_references_dataset])

    not_nan = 0
    anan = 0
    for i, r in issues_df.iterrows():
        if r['cve_id'].strip() != 'nan':
            not_nan +=1
        else:
            anan += 1
            
    print(not_nan)
    print(anan)
    print(issues_df.shape)
    print(not_nan/issues_df.shape[0])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

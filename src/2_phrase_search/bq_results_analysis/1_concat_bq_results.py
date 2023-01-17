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

import pandas as pd
import gzip

from src.utils.utils import get_files_in_from_directory

bq_results_directory = 'results/bigquery_results'

files = get_files_in_from_directory(bq_results_directory)
issues_per_filename = []
sec_issues_per_filename = []


dfs_all = []
dfs = []
for filename in files:
    with gzip.open(filename) as f:
        temp_df = pd.read_csv(f)
        dfs_all.append(temp_df)
        with_keywords = temp_df[temp_df['found_keywords'].notnull()]
        issues_per_filename.append({
            'filename': filename,
            'number_of_issues':temp_df.shape[0]
        })
        sec_issues_per_filename.append({
            'filename': filename,
            'number_of_issues':with_keywords.shape[0]
        })
        dfs.append(with_keywords)

df = pd.concat(dfs)
df.to_csv('issues_witf_sec_keywords.csv', index=False)
dfx = pd.concat(dfs_all)
dfx.to_csv('all_issues_scanned.csv', index=False)
print('ok')


temp_df = pd.DataFrame(issues_per_filename)
temp_df.to_csv('issues_per_file.csv', index=False)
temp_df = pd.DataFrame(sec_issues_per_filename)
temp_df.to_csv('sec_issues_per_file.csv', index=False)
print('Read n files', len(dfs))
print('ok')


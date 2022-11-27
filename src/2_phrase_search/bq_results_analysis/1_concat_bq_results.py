#!/usr/bin/env python3
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


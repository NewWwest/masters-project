#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import pandas as pd
from src.utils.constants import max_allowed_commits_in_link


data_files = [
    'data\\most_starred\\filterred_commits.csv',
    'data\\most_used_mvn\\filterred_commits.csv',
    'data\\most_used_npm\\filterred_commits.csv',
    'data\\most_used_pypi\\filterred_commits.csv'
]

dfs = [pd.read_csv(data_file) for data_file in data_files]
refs = pd.concat(dfs)

by_issue = {}
for i, ref in refs.iterrows():
    if ref['reference_type'] != 'ref_commit':
        continue

    if ref['reference_value'].startswith('https://'):
        sha = ref['reference_value'].split('/')[-1]
    else:
        sha = ref['reference_value']
    res = {
        'report_id':ref['issue_url'],
        'repo_owner':ref['repo_full_name'].split('/')[0],
        'repo_name':ref['repo_full_name'].split('/')[1],
        'commit_sha':sha
    }

    if ref['issue_url'] not in by_issue:
        by_issue[ref['issue_url']] = []
    by_issue[ref['issue_url']].append(res)

result_data = []
for issue_url, refs in by_issue.items():
    if len(refs) > max_allowed_commits_in_link:
        continue

    result_data += refs

df = pd.DataFrame(result_data)
df.sort_values('report_id', inplace=True)
df.to_csv('test_extracted_commits.csv', index=False)

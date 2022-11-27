#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import pandas as pd

input_file = r'issues_witf_sec_keywords.csv'

df = pd.read_csv(input_file)
result = []
issues_part = df[df.apply(lambda x: pd.isna(x['commit_url']), axis=1)]


issues_part_groups = issues_part.groupby('issue_url')
for kk, g in issues_part_groups:
    entity_urls = ';'.join(g['entity_urls'])
    entity_urls = ';'.join(list(set(entity_urls.split(';'))))
    found_keywords = ';'.join(g['found_keywords'])
    found_keywords = ';'.join(list(set(found_keywords.split(';'))))
    temp = {
        'commit_url': float('nan'),
        'issue_url': kk,
        'found_keywords': found_keywords,
        'entity_urls': entity_urls,
    }
    result.append(temp)

commits_part = df[df.apply(lambda x: pd.isna(x['issue_url']), axis=1)]
commits_part_groups = commits_part.groupby('commit_url')
for kk, g in commits_part_groups:
    entity_urls = ';'.join(g['entity_urls'])
    entity_urls = ';'.join(list(set(entity_urls.split(';'))))
    found_keywords = ';'.join(g['found_keywords'])
    found_keywords = ';'.join(list(set(found_keywords.split(';'))))
    temp = {
        'commit_url': kk,
        'issue_url': float('nan'),
        'found_keywords': found_keywords,
        'entity_urls': entity_urls,
    }
    result.append(temp)


issues_df = pd.DataFrame(result)
issues_df.to_csv('results_from_bq_search.csv', index=False)
print('ok')

# print(x['commit_url'])
# print(x['issue_url'])
# print(x['found_keywords'])
# print(x['entity_urls'])
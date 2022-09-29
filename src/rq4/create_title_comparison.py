import json
from src.loaders.temp.OsvReportsLoader import OsvReportsLoader
import time
import requests
import pandas as pd

vulnerabilities_path  ='/Users/awestfalewicz/Private/data/advisory-database/advisories'
path='/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/src/linked_commit_investigation/output_path.json.cache'
vulnerabilities = OsvReportsLoader().load(vulnerabilities_path)

all_commits = pd.read_csv('/Users/awestfalewicz/Projects/all_commit_data.csv')
commits_by_cve = all_commits.groupby('cve_id').apply(list)
with open(path, "r") as f:
    mapping = json.load(f)


all_commit_data = []
title_mapping = []

for x in mapping:
    if len(mapping[x])>0:
        report_info = vulnerabilities.reports_raw_data[x]
        if 'summary' not in report_info:
            continue
        if len(mapping[x]) > 10:
            continue

        print(x)
        commits_info = commits_by_cve[x]
        for index, commit in commits_info.iterrows():
            commit_data = json.loads(commit['commit'])
            print(commit)
            title_mapping.append({'cve_id': x,'cve_title':report_info['summary'], 'commit_msg': commit_data['commit']['message'], 'matches':'XXX'})


df1 = pd.DataFrame(title_mapping)
df1 = df1.sample(100)
df1.to_csv('title_mapping.csv', index=False)
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

input_data_location = 'results/checkpoints_fixMapper'

input_issues_references_dataset = [
    r'data\most_starred\filterred_commits.csv',
    r'data\most_used_npm\filterred_commits.csv',
    r'data\most_used_pypi\filterred_commits.csv',
    r'data\most_used_mvn\filterred_commits.csv',
]
input_issues_content_dataset = [
    r'data\most_starred\filterred_issues.csv',
    r'data\most_used_npm\filterred_issues.csv',
    r'data\most_used_pypi\filterred_issues.csv',
    r'data\most_used_mvn\filterred_issues.csv',
]


def standarize_found_refs():
    issues_df = pd.concat([pd.read_csv(x) for x in input_issues_content_dataset])
    issues_refs_df = pd.concat([pd.read_csv(x) for x in input_issues_references_dataset])
    issues_refs_dict = dict(tuple(issues_refs_df.groupby('issue_url')))

    res = {}
    for i, r in issues_df.iterrows():
        res[r['issue_url']] = []
        segments = r['repo_full_name'].split('/')
        segments_url = r['issue_url'].split('/')

        res[r['issue_url']].append({
            'repo_owner': segments[0],
            'repo_name': segments[0],
            'ref_type': 'issue',
            'ref_value': segments_url[-1],
        })
        
        if r['issue_url'] not in issues_refs_dict:
            continue

        for _, asd in issues_refs_dict[r['issue_url']].iterrows():
            if asd['reference_type'] == 'ref_issue':
                ref_segments = asd['reference_value'].split('/')
                res[r['issue_url']].append({
                    'repo_owner': ref_segments[4],
                    'repo_name': ref_segments[5],
                    'ref_type': 'issue',
                    'ref_value': ref_segments[7],
                })
            elif asd['reference_type'] == 'ref_commit':
                if asd['reference_value'].startswith('http'):
                    ref_segments = asd['reference_value'].split('/')
                    res[r['issue_url']].append({
                        'repo_owner': ref_segments[4],
                        'repo_name': ref_segments[5],
                        'ref_type': 'commit',
                        'ref_value': ref_segments[7],
                    })
                else:
                    res[r['issue_url']].append({
                        'repo_owner': segments[0],
                        'repo_name': segments[1],
                        'ref_type': 'commit',
                        'ref_value': asd['reference_value'],
                    })

    return res


def load_vuln_refs():
    res = []
    data_files = get_files_in_from_directory(input_data_location, '.json')
    for f in data_files:
        with open(f, 'r') as f_temp:
            data = json.load(f_temp)
        for x in data:
            res += data[x]
            
    return res


def convert_to_dicts(refs):
    refs_by_repo = {}
    for r in refs:
        for x in refs[r]:
            repo_full_name = f'{x["repo_owner"]}/{x["repo_name"]}'
            if repo_full_name not in refs_by_repo:
                refs_by_repo[repo_full_name] = []
            refs_by_repo[repo_full_name].append(x)
    
    return refs_by_repo


def main():
    issue_refs = standarize_found_refs()
    vuln_refs = load_vuln_refs()
    
    vuln_refs_by_repo = {}
    for x in vuln_refs:
        repo_full_name = f'{x["repo_owner"]}/{x["repo_name"]}'
        if repo_full_name not in vuln_refs_by_repo:
            vuln_refs_by_repo[repo_full_name] = []
        vuln_refs_by_repo[repo_full_name].append(x)

    same_repos= set()
    mapped = set()
    for issue in issue_refs:
        for issue_ref in issue_refs[issue]:
            repo_full_name = f'{x["repo_owner"]}/{x["repo_name"]}'
            if repo_full_name in vuln_refs_by_repo:
                same_repos.add(repo_full_name)
                for vuln_refx in  vuln_refs_by_repo[repo_full_name]:
                    if vuln_refx['reference_type'] == issue_ref['ref_type'] and vuln_refx['reference_type'] == 'issue':
                        if vuln_refx['reference_value'] == issue_ref['ref_value']:
                            mapped.add(issue)
                            
                    if vuln_refx['reference_type'] == issue_ref['ref_type'] and vuln_refx['reference_type'] == 'commit':
                        if str(vuln_refx['reference_value'][:10]) == str(issue_ref['ref_value'][:10]):
                            mapped.add(issue)

    print('Same repos:', len(same_repos), same_repos)
    print('Same refer:', len(mapped), mapped)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

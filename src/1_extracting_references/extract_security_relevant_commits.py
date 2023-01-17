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
import tqdm

import src.utils.utils as utils
import src.utils.constants as constants
from src.mining.IssuesMiners.extract_commits_from_issue import extract_commits_from_issue

# Either the directory containing the single data_full.json file
# or the directory containing the data_<report_id> files (as in reproduction package)
mapper_results_directory = 'path_to_downloaded_references'
result_file = 'security_related_commits_in_vuln.csv'


def make_csv_object(report_id, repo_owner, repo_name, commit_sha):
    res = {}
    res['report_id']  = report_id
    res['repo_owner'] = repo_owner
    res['repo_name']  = repo_name
    res['commit_sha'] = commit_sha
    return res
    

def main():
    data_files = utils.get_files_in_from_directory(mapper_results_directory, '.json')
    result = []
    with tqdm.tqdm(total=len(data_files)) as pbar:
        for data_file in data_files:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            for report_id in data:
                for ref in data[report_id]:
                    ref_aliases = json.loads(ref['aliases'])

                    if ref['reference_type'] == 'commit':
                        ref_data = json.loads(ref['data'])
                        for alias_id in ref_aliases:
                            result.append(make_csv_object(alias_id, ref['repo_owner'], ref['repo_name'], ref_data['sha']))
                    elif ref['reference_type'] == 'issue':
                        ref_data = json.loads(ref['data'])
                        commits = extract_commits_from_issue(ref['repo_owner'], ref['repo_name'], ref_data)
                        if len(commits) < constants.max_allowed_commits_in_link:
                            for c in commits:
                                for alias_id in ref_aliases: 
                                    result.append(make_csv_object(alias_id, ref['repo_owner'], ref['repo_name'], c))
                    elif ref['reference_type'] == 'compare':
                        ref_data = json.loads(ref['data'])
                        commits = [x['sha'] for x in ref_data['commits']]
                        if len(commits) <= constants.max_allowed_commits_in_link:
                            for c in commits:
                                for alias_id in ref_aliases: 
                                    result.append(make_csv_object(alias_id, ref['repo_owner'], ref['repo_name'], c))
                    else:
                        raise Exception(f'Unknown reference_type {ref["reference_type"]}')

            pbar.update()

    result_df = pd.DataFrame(result)
    result_df.drop_duplicates()
    result_df.to_csv(result_file, index = False)


if __name__ == '__main__':
    main()
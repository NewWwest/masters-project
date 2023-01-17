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
import json
import pandas as pd

import src.utils.utils as utils

mapper_results_directory = 'results/checkpoints_fixMapper'
p = 'data/rq2_precission/data/most_used_pypi'
input_issue_references_dataset = f'{p}/filterred_commits.csv'
input_issue_content_dataset = f'{p}/filterred_issues.csv'


def cross_referenced_from_issue(report_id, repo_owner, repo_name, issue_data):
    result = []
    cross_referenced_issues = [x['source']['issue'] for x in issue_data['timeline_data'] if x['event']=='cross-referenced']
    for cross_ref in cross_referenced_issues:
        temp_res = {
            'id':report_id,
            'repo_owner': repo_owner,
            'repo_name': repo_name,
            'reference_type': 'issue',
            'reference_value': cross_ref['number']
        }
        result.append(temp_res)

    referenced_commits = [x for x in issue_data['timeline_data'] if 'commit_id' in x and x['commit_id'] != None]
    for c in referenced_commits:
        if 'html_url' in c:
            seg = c['html_url'].split('/')
            ref_repo_owner = seg[3]
            ref_repo_name = seg[4]
        else:
            seg = c['commit_url'].split('/')
            ref_repo_owner = seg[4]
            ref_repo_name = seg[5]

        if repo_owner == ref_repo_owner and repo_name == ref_repo_name:
            temp_res = {
                'id':report_id,
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'reference_type': 'commit',
                'reference_value': c['commit_id'][0:10]
            }
            result.append(temp_res)

    if 'pull_request_commits' in issue_data:
        for c in issue_data['pull_request_commits']:
            temp_res = {
                'id':report_id,
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'reference_type': 'commit',
                'reference_value': c['sha'][0:10]
            }
            result.append(temp_res)

    return result

def fill_df_with_refs(dictionary, df):
    for i, ref in df.iterrows():
        repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
        if repo_full_name not in dictionary:
            dictionary[repo_full_name]=[]

        dictionary[repo_full_name].append(ref)


def load_references_in_vulnerabilities():
    data_files = utils.get_files_in_from_directory(mapper_results_directory, '.json')
    result = []
    cross_references = []
    for data_file in data_files:
        print(f'Processing... {data_file}')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for report_id in data:
            for ref in data[report_id]:
                if ref['reference_type'] == 'commit':
                    ref_data = json.loads(ref['data'])
                    temp_res = {
                        'id':report_id,
                        'repo_owner': ref['repo_owner'],
                        'repo_name': ref['repo_name'],
                        'reference_type': ref['reference_type'],
                        'reference_value': ref_data['sha'][0:10]
                    }
                    result.append(temp_res)
                elif ref['reference_type'] == 'issue':
                    ref_data = json.loads(ref['data'])
                    temp_res = {
                        'id':report_id,
                        'repo_owner': ref['repo_owner'],
                        'repo_name': ref['repo_name'],
                        'reference_type': ref['reference_type'],
                        'reference_value': ref_data['number']
                    }
                    result.append(temp_res)
                    cross_references_temp = cross_referenced_from_issue(report_id, ref['repo_owner'],ref['repo_name'], ref_data)
                    cross_references+=cross_references_temp
                elif ref['reference_type'] == 'compare':
                    ref_data = json.loads(ref['data'])
                    temp_res = {
                        'id':report_id,
                        'repo_owner': ref['repo_owner'],
                        'repo_name': ref['repo_name'],
                        'reference_type': ref['reference_type'],
                        'reference_value': ref['reference_value']
                    }
                    result.append(temp_res)
                    if 'commits' in ref_data:
                        for commit in ref_data['commits']:
                            temp_res = {
                                'id':report_id,
                                'repo_owner': ref['repo_owner'],
                                'repo_name': ref['repo_name'],
                                'reference_type': 'commit',
                                'reference_value': commit['sha']
                            }
                            result.append(temp_res)
                else:
                    raise Exception(f'Unknown reference_type {ref["reference_type"]}')

    result_df = pd.DataFrame(result)
    result_df.drop_duplicates(inplace=True)

    cross_references_result_df = pd.DataFrame(cross_references_temp)
    cross_references_result_df.drop_duplicates(inplace=True)

    refs_by_repo ={}
    fill_df_with_refs(refs_by_repo, result_df)

    only_cross_references ={}
    fill_df_with_refs(only_cross_references, result_df)
    fill_df_with_refs(only_cross_references, cross_references_result_df)

    return refs_by_repo, only_cross_references

def extract_refs_from_issue(repo_full_name, issue_obj, refs):
    result =[]

    repo_owner = repo_full_name.split('/')[0]
    repo_name = repo_full_name.split('/')[1]
    direct_res = {
        'repo_owner': repo_owner,
        'repo_name': repo_name,
        'reference_type': 'issue',
        'reference_value': issue_obj['number']
    }
    result.append(direct_res)
    for i,ref in refs.iterrows():
        if ref['reference_type'] == 'ref_commit':
            res_temp = {
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'reference_type': 'commit',
                'reference_value': ref['reference_value'][0:10]
            }
            result.append(res_temp)
        elif ref['reference_type'] == 'ref_issue':
            res_temp = {
                'repo_owner': repo_owner,
                'repo_name': repo_name,
                'reference_type': 'issue',
                'reference_value': ref['reference_value']
            }
            result.append(res_temp)
        else:
            raise Exception(f'Invalid reference_type {ref["reference_type"]}')

    return result


def is_same_ref(left, right):
    return left['repo_owner'] == right['repo_owner'] and \
        left['repo_name'] == right['repo_name'] and \
        left['reference_type'] == right['reference_type'] and \
        left['reference_value'] == right['reference_value']


def main():
    direct_ref_in_vul, indirect_ref_in_vul = load_references_in_vulnerabilities()
    issues = pd.read_csv(input_issue_content_dataset)
    references = pd.read_csv(input_issue_references_dataset)

    refs_by_issue = {}
    direct_mapping = {}
    indirect_mapping = {}
    refs_by_issue_df = references.groupby('issue_url')
    for issue_url, refs_df in refs_by_issue_df:
        refs_by_issue[issue_url] = refs_df

    for i, issue in issues.iterrows():
        direct_mapping[issue['issue_url']] = []
        indirect_mapping[issue['issue_url']] = []
        if issue['repo_full_name'] not in direct_ref_in_vul and issue['repo_full_name'] not in indirect_ref_in_vul:
            continue

        if issue['issue_url'] in refs_by_issue:
            refs_of_issue = refs_by_issue[issue['issue_url']]
        else:
            refs_of_issue = []

        print('Processing...', issue['issue_url'])
        normalized_refs_in_issues = extract_refs_from_issue(issue['repo_full_name'], json.loads(issue['full_issue']), refs_of_issue)

        if issue['repo_full_name'] in direct_ref_in_vul:
            normalized_refs_in_vulnerabilities = direct_ref_in_vul[issue['repo_full_name']]
            for left in normalized_refs_in_issues:
                for right in normalized_refs_in_vulnerabilities:
                    if is_same_ref(left, right):
                        direct_mapping[issue['issue_url']].append(right['id'])
                        indirect_mapping[issue['issue_url']].append(right['id'])

        if issue['repo_full_name'] in indirect_ref_in_vul:
            normalized_refs_in_vulnerabilities = indirect_ref_in_vul[issue['repo_full_name']]
            for left in normalized_refs_in_issues:
                for right in normalized_refs_in_vulnerabilities:
                    if is_same_ref(left, right):
                        indirect_mapping[issue['issue_url']].append(right['id'])

    with open('direct_vulnerabilities_mapping.json', 'w') as f:
        json.dump(direct_mapping, f, indent=2)
    with open('indirect_vulnerabilities_mapping.json', 'w') as f:
        json.dump(indirect_mapping, f, indent=2)

    print('All issues', len(direct_mapping), len(indirect_mapping))
    print('Counting only more direct references:', len([1 for x in direct_mapping if len(direct_mapping[x])>0]))
    print('Counting more references:', len([1 for x in indirect_mapping if len(indirect_mapping[x])>0]))




if __name__ == '__main__':
    main()

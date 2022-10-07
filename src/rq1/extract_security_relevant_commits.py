import sys
sys.path.insert(0, r'D:\Projects\aaa')

import json
import pandas as pd
import tqdm

import src.utils.utils as utils
import src.utils.constants as constants
from src.mining.IssuesMiners.extract_commits_from_issue import extract_commits_from_issue

mapper_results_directory = 'results/checkpoints_fixMapper'
result_file = 'results/security_related_commits_in_vuln.csv'


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
                            result.append(make_csv_object(alias_id, ref['repo_owner'],ref['repo_name'], ref_data['sha']))
                    elif ref['reference_type'] == 'issue':
                        ref_data = json.loads(ref['data'])
                        commits = extract_commits_from_issue(ref['repo_owner'], ref['repo_name'], ref_data)
                        if len(commits) < constants.max_allowed_commits_in_link:
                            for c in commits:
                                for alias_id in ref_aliases: 
                                    result.append(make_csv_object(alias_id, ref['repo_owner'],ref['repo_name'], c))
                    elif ref['reference_type'] == 'compare':
                        ref_data = json.loads(ref['data'])
                        commits = [x['sha'] for x in ref_data['commits']]
                        if len(commits) <= constants.max_allowed_commits_in_link:
                            for c in commits:
                                for alias_id in ref_aliases: 
                                    result.append(make_csv_object(alias_id, ref['repo_owner'],ref['repo_name'], c))
                    else:
                        raise Exception(f'Unknown reference_type {ref["reference_type"]}')

            pbar.update()

    result_df = pd.DataFrame(result)
    result_df.drop_duplicates()
    result_df.to_csv(result_file, index = False)


if __name__ == '__main__':
    main()
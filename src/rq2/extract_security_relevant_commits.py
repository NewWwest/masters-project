import pandas as pd
import src.utils.constants as constants

data_files = [
    'src/src2/rq2_precission/data/most_starred/refs_with_commits.csv',
    'src/src2/rq2_precission/data/most_used_mvn/refs_with_commits.csv',
    'src/src2/rq2_precission/data/most_used_npm/refs_with_commits.csv',
    'src/src2/rq2_precission/data/most_used_pypi/refs_with_commits.csv'
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
    if len(refs) > constants.max_allowed_commits_in_link:
        continue

    result_data += refs

df = pd.DataFrame(result_data)
df.sort_values('report_id', inplace=True)
df.to_csv('test_extracted_commits.csv', index=False)
# repo_id,repo_full_name,issue_id,issue_url,candidate_commit,reference_type,reference_value,commit_data,commit_reachable
# report_id,repo_owner,repo_name,commit_sha

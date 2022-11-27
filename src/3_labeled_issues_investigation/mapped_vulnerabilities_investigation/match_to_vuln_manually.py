#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json
import time
import pandas as pd
import requests
from src.secrets import nvd_token
import json
import pandas as pd

allowed_distance = 3
mapper_results_directory = 'results/checkpoints_fixMapper'
nvd_by_keyword ='https://services.nvd.nist.gov/rest/json/cves/1.0?keyword={keyword}&resultsPerPage=2000&apiKey={nvd_token}'
nvd_by_cpe ='https://services.nvd.nist.gov/rest/json/cves/1.0?cpeName={cpeName}&resultsPerPage=2000&apiKey={nvd_token}'

p = 'src/src2/rq2_precission/data/most_starred'
input_issue_references_dataset = f'{p}/filterred_commits.csv'
input_issue_content_dataset = f'{p}/filterred_issues.csv'
input_annoted_issues = f'{p}/annotated_issues.csv'

input_repos_to_cpes = f'{p}/repos_to_cpes.csv'
input_issues_to_cves = f'{p}/manually_annotated_mapping.todo.csv'

def load_issues_df():
    issues_df = pd.read_csv(input_issue_content_dataset)
    annotated_issues_df = pd.read_csv(input_annoted_issues)
    mapx = {}
    for i, annoted_issue in annotated_issues_df.iterrows():
        mapx[annoted_issue['issue_url']] = str(annoted_issue['is_actual_fix']).strip().startswith('valid')
    issues_df = issues_df[issues_df.apply(lambda x: mapx[x['issue_url']] if x['issue_url'] in mapx else False, axis=1)]
    return issues_df


def find_cpes_for_repos():
    repos_to_cpes = []
    issues_df = load_issues_df()

    by_repo = issues_df.groupby('repo_full_name')
    for k, g in by_repo:
        time.sleep(1)
        segments = k.split('/')
        nvd_url = nvd_by_keyword.replace('{keyword}', segments[1]).replace('{nvd_token}', nvd_token)
        resp = requests.get(nvd_url)
        cves = resp.json()
        cpe_ids = []
        for cve in cves['result']['CVE_Items']:
            cpe_matches = [x['cpe_match'] for x in cve['configurations']['nodes']]
            # [item for sublist in l for item in sublist]
            cpe_ids += [y['cpe23Uri'] for x in cpe_matches for y in x]

        cpe_ids = list(set([f'{cpe.split(":")[3]}:{cpe.split(":")[4]}' for cpe in cpe_ids]))

        res = {}
        res['repo_full_name'] = k
        res['cpe_ids'] = json.dumps(cpe_ids)
        repos_to_cpes.append(res)


    df = pd.DataFrame(repos_to_cpes)
    df.to_csv(input_repos_to_cpes, index=False)


def get_vulnerabilties_for_repos():
    issues_df = load_issues_df()
    by_repo = issues_df.groupby('repo_full_name')

    repos_to_cpes = pd.read_csv(input_repos_to_cpes)
    vulnerabilities_per_repo = []

    for k, g in by_repo:
        row = repos_to_cpes.loc[repos_to_cpes['repo_full_name'] == k]
        if row['cpe_ids'].shape[0] < 1:
            res = {}
            res['repo_full_name'] = k
            res['cve_ids'] = json.dumps([])
            res['cve_objects'] = json.dumps([])
            vulnerabilities_per_repo.append(res)
            continue
        cpes_json = row['cpe_ids'].iloc[0]
        cpes = json.loads(cpes_json)
        cve_objects = []
        for cpe in cpes:
            time.sleep(0.5)
            full_cpe=f'cpe:2.3:a:{cpe}'
            full_cpe = full_cpe.replace('\\\\','\\')
            nvd_url = nvd_by_cpe.replace('{cpeName}', full_cpe).replace('{nvd_token}', nvd_token)
            resp = requests.get(nvd_url)
            cves = resp.json()
            for cve in cves['result']['CVE_Items']:
                cve_objects.append(cve)
        
        res = {}
        res['repo_full_name'] = k
        res['cve_ids'] = json.dumps([x['cve']['CVE_data_meta']['ID'] for x in cve_objects])
        res['cve_objects'] = json.dumps(cve_objects)
        vulnerabilities_per_repo.append(res)

    df = pd.DataFrame(vulnerabilities_per_repo)
    return df


def create_nvd_mapping_file():
    cves_to_repos = get_vulnerabilties_for_repos()

    issues_df = load_issues_df()
    by_repo = issues_df.groupby('repo_full_name')

    vulnerabilities_per_issue = []
    for k, x in by_repo:
        row = cves_to_repos.loc[cves_to_repos['repo_full_name'] == k]
        if row.shape[0] < 1:
            continue
        cves = json.loads(row['cve_objects'].iloc[0])
        for i, issue in x.iterrows():
            res = {}
            full_issue = json.loads(issue['full_issue'])
            res['issue_url'] = issue['issue_url']
            res['issue_title'] = full_issue['title']
            res['cve_id']=' XXX '
            vulnerabilities_per_issue.append(res)
        for cve in cves:
            res = {}
            res['issue_url']='         '
            res['issue_title']=cve['cve']['description']['description_data'][0]['value']
            res['cve_id']=cve['cve']['CVE_data_meta']['ID']
            vulnerabilities_per_issue.append(res)
        res = {}
        res['issue_url']='q'
        res['issue_title']='q'
        res['cve_id']='q'
        vulnerabilities_per_issue.append(res)

            
    df = pd.DataFrame(vulnerabilities_per_issue)
    df.to_csv(input_issues_to_cves, index=False)



if __name__ == '__main__':
    # # Step 1: Generate the file with found cpes
    # find_cpes_for_repos()

    # # Step 2: Manually filter the file to find cpes that may relate to the repo

    # # Step 3: Get vulnerabilities for the repos and create the file for review
    # create_nvd_mapping_file()

    # # Step 4: Find similar packages in OSV
    pass
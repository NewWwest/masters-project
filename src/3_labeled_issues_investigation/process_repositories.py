#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json
import pandas as pd
import os

from src.mining.IssuesMiners.IssueDownloader import IssueDownloader
from src.mining.IssuesMiners.IssueFinder import IssueFinder
from src.proxies.GitHubProxy import GithubProxy


files_to_ignore = ['dockerfile', 'gemfile']
extensions_to_ignore = ['.md', '.json' , '.txt', '.gradle', '.sha', '.lock', '.ruby-version', '.yml', '.yaml', '.xml', '.xaml']

github_proxy = GithubProxy()


def filter_issues(issues_data_path, commits_data_path, issues_result_path, commits_result_path):
    with open(issues_data_path, 'r') as f:
        issues_data = json.load(f)
    with open(commits_data_path, 'r') as f:
        commits_data = json.load(f)
    issue_flag_map = {}
    for i in issues_data:
        issue_flag_map[i['url']] = False

    for c in commits_data:
        commit_data = c['commit_data']
        for file in commit_data['files']:
            filename = os.path.basename(file['filename']).lower()
            file_like_path, file_extension = os.path.splitext(file['filename'])

            if filename not in files_to_ignore and file_extension.lower() not in extensions_to_ignore:
                issue_flag_map[c['issue_url']]=True
                break
            

    filterred_issues = [i for i in issues_data if issue_flag_map[i['url']]]
    filterred_commits = [c for c in commits_data if issue_flag_map[c['issue_url']]]

    with open(issues_result_path, 'w') as f:
        json.dump(filterred_issues, f)
    
    with open(commits_result_path, 'w') as f:
        json.dump(filterred_commits, f)


def create_review_csv(issues_input, output_file):
    issues = pd.read_csv(issues_input)
    result = []

    for index, issue in issues.iterrows():
        data = {}
        data['repo_full_name'] = issue['repo_full_name']
        data['issue_url'] = issue['issue_url']

        json_data = json.loads(issue['full_issue'])
        data['html_url'] = ' ' + json_data['html_url'] + ' '
        data['issue_title'] = json_data['title']
        data['is_actual_fix'] = ' XXX '
        result.append(data)
        
    to_save = pd.DataFrame(result)
    to_save = to_save.sample(n=150)
    to_save = to_save.sort_values('repo_full_name')
    to_save.to_csv(output_file, index=False)


if __name__ == '__main__':
    px = 'src/rq2_precission/final_results/test'
    repos_path = f'{px}/top_starred_repositories.csv'
    ifx = IssueFinder('results/issue_finder_checkpoints')
    idx = IssueDownloader('results/issue_finder_checkpoints')

    ifx.find_issues(repos_path, ['secur', 'vulnerab', 'exploit'], f'{px}/repositories_with_sec_label.csv', f'{px}/issue_list.csv')
    idx.download_issues(f'{px}/issue_list.csv', f'{px}/issue_content.json', f'{px}/commit_content.csv')
    filter_issues(f'{px}/issue_content.json', f'{px}/commit_content.csv', f'{px}/filtered_issue_content.json', f'{px}/filtered_commit_content.json')

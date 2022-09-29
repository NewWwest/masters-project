import urllib.parse
import json
import pandas as pd
import os
import os

from src.proxies.GitHubProxy import GithubProxy
from src.utils.constants import github_api

class IssueFinder:
    def __init__(self, checkpoint_directory) -> None:
        self.checkpoint_directory = checkpoint_directory
        self.github_proxy = GithubProxy()


    def _search_for_label(self, repo_id, keyword):
        q = urllib.parse.quote_plus(keyword)
        repo = urllib.parse.quote_plus(str(repo_id))
        labels_endpoint = f'{github_api}/search/labels?q={q}&repository_id={repo}'
        labels = self.github_proxy.iterate_search_endpoint(labels_endpoint, None)
        if labels == None:
            return []
        else:
            return [label['name'] for label in labels]


    def _get_security_labels(self, repos, issue_label_keywords, outputFile):
        for i, r in repos.iterrows():
            repo_id = str(int(r['id']))
            labels = []
            for issue_label_keyword in issue_label_keywords:
                labels += self._search_for_label(repo_id, issue_label_keyword)

            repos.at[i,'labels'] = json.dumps(labels)

            if len(labels) > 0:
                repos.to_csv(f'{self.checkpoint_directory}/labels_{repo_id}.csv', index=False)

        repos_with_labels = repos[repos.apply(lambda x: str(x['labels']) != '[]', axis=1)]
        repos_with_labels.to_csv(outputFile, index=False)
        return repos_with_labels

    
    def _find_issues_in_repos(self, repos, issues_result_file): 
        issues_for_rest_repos = []
        for i, repo in repos.iterrows():
            if repo['labels'] == None or repo['labels'] == '':
                continue

            labels = json.loads(repo['labels'])
            if len(labels) == 0:
                continue

            issues = []
            for l in labels:
                i = self._get_issues_for_label(repo['full_name'], repo['id'], l)
                issues += i

            if len(issues) > 0:
                issues_df = pd.DataFrame(issues)
                issues_df.to_csv(f'{self.checkpoint_directory}/issues_{int(repo["id"])}.csv')
                issues_for_rest_repos.append(issues_df)

        df = pd.concat(issues_for_rest_repos)
        df.to_csv(issues_result_file, index=False)
        return df


    def _get_issues_for_label(self, repo_full_name, repo_id, label):
        q = urllib.parse.quote_plus(f'repo:{repo_full_name} state:closed label:{label}')	
        issues_endpoint = f'{github_api}/search/issues?q={q}&sort=created&order=desc' 
        issues = self.github_proxy.iterate_search_endpoint(issues_endpoint, None)
        result = []
        for issue in issues:
            result.append({
                'repo_id':repo_id,
                'repo_full_name':repo_full_name,
                'issue_id':issue['id'],
                'issue_url':issue['url']
                })
        
        return result


    # issue_label_keywords = ['secur', 'vulnerab', 'exploit']
    def find_issues(self, path_to_repos_csv, issue_label_keywords, labels_result_file, issues_result_file):
        repos = pd.read_csv(path_to_repos_csv)
        repos_with_labels = self._get_security_labels(repos, issue_label_keywords, labels_result_file)
        issues_for_rest_repos = self._find_issues_in_repos(repos_with_labels, issues_result_file)
        # # for reading the isses from the hard drive
        # issues_for_rest_repos = pd.concat([pd.read_csv(f) for f in src.utils.utils.get_files_in_from_directory('.csv','issues_')]
        return issues_for_rest_repos



if __name__ == '__main__':
    finder = IssueFinder('results/issue_finder_checkpoints')
    path_to_repos_csv = r'src\rq2_precission\final_results\most_starred\top_starred_repositories.csv'
    finder.find_issues(path_to_repos_csv, ['secur', 'vulnerab', 'exploit'], 'repos_with_security_labels.csv', 'security_issues.csv')
    
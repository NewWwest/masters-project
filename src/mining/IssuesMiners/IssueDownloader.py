import json
import pandas as pd

from src.mining.IssuesMiners.extract_commits_from_issue import extract_commits_from_issue
from src.proxies.GitHubProxy import GithubProxy

checkpoint_frequency = 100

class IssueDownloader:
    def __init__(self, checkpoint_directory) -> None:
        self.checkpoint_directory = checkpoint_directory
        self.github_proxy = GithubProxy()
        pass

    def download_issues(self, path_to_issues_csv, issue_info_output, commit_info_output):
        all_issue_data = self._download_issue_data(path_to_issues_csv, issue_info_output)
        all_commit_data = self._download_commits_data(all_issue_data, commit_info_output)


    def _download_commits_data(self, all_issue_data, commit_info_output):
        # with open('issue_info_output', 'r') as f:
        #     all_issue_data = json.load(f)

        all_comits_data = []
        current_comits_data = []
        for issue in all_issue_data:
            segments = issue['url'].split('/')
            commits = extract_commits_from_issue(segments[4], segments[5], issue)
            for hash in commits:
                commit_info = self.github_proxy.get_commit_data(segments[4], segments[5], hash)
                if commit_info != None:
                    res = { 
                        'repo_full_name': f'{segments[4]}/{segments[5]}',
                        'issue_id': issue['id'],
                        'issue_url': issue['url'],
                        'commit_sha': hash, 
                        'commit_data':commit_info
                    }
                    all_comits_data.append(res)
                    current_comits_data.append(res)

            if len(current_comits_data) >= checkpoint_frequency:
                with open(f'{self.checkpoint_directory}/full_commits_{segments[4]}-{segments[5]}-{issue["id"]}.json', 'w') as f:
                    json.dump(current_comits_data, f)
                current_comits_data = []

        with open(f'{self.checkpoint_directory}/full_commits_last.json', 'w') as f:
            json.dump(current_comits_data, f)
        current_comits_data = []
        with open(commit_info_output, 'w') as f:
            json.dump(all_comits_data, f)
        return all_comits_data


    def _download_issue_data(self, input_path, output_path):
        issues = pd.read_csv(input_path)
        all_issue_data = []
        current_issues_data = []

        for index, i in issues.iterrows():
            segments = i['repo_full_name'].split('/')
            issue_number = i['issue_url'].split('/')[-1]
            issue_data = self.github_proxy.get_issue_data(segments[0], segments[1], issue_number)
            if issue_data == None:
                continue
            all_issue_data.append(issue_data)
            current_issues_data.append(issue_data)

            if len(current_issues_data) >= checkpoint_frequency:
                with open(f'{self.checkpoint_directory}/full_issues_{index}.json', 'w') as f:
                    json.dump(current_issues_data, f)
                current_issues_data = []
        
        
        with open(f'{self.checkpoint_directory}/full_issues_last.json', 'w') as f:
            json.dump(current_issues_data, f)
        current_issues_data = []

        with open(output_path, 'w') as f:
            json.dump(all_issue_data, f)

        return all_issue_data

        
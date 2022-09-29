import json
import pandas as pd 

features_calculated_cache_location = 'results\checkpoints3'
features_selected_cache_location = 'results\checkpoints4'
fix_to_norm_ratio = 10
class CommitSelectorCSV:
    def __init__(self, mapping_file) -> None:
        mapping_csv = pd.read_csv(mapping_file)

        self.commits = {}
        for i,r in mapping_csv.iterrows():
            repo_full_name = f'{r["repo_owner"]}/{r["repo_name"]}'
            if repo_full_name not in self.commits:
                self.commits[repo_full_name] = []
            
            self.commits[repo_full_name].append(r)


    def get_repos_with_atleastn_fix_commits(self, n):
        with_fix_count = []
        for x in self.commits:
            if len(self.commits[x]) >= n:
                with_fix_count.append((x, len(self.commits[x])))

        with_fix_count = sorted(with_fix_count, key=lambda x: x[1], reverse=True)
        return [x[0] for x in with_fix_count]

    def shas_for_repo(self, repo_full_name):
        if repo_full_name not in self.commits:
            return []

        return list([x['commit_sha'] for x in self.commits[repo_full_name]])

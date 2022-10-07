import pandas as pd 

class CommitProvider:
    def __init__(self, mapping_file) -> None:
        mapping_csv = pd.read_csv(mapping_file)
        self.commits = {}
        for i,r in mapping_csv.iterrows():
            repo_full_name = f'{r["repo_owner"]}/{r["repo_name"]}'
            if repo_full_name not in self.commits:
                self.commits[repo_full_name] = set()
            
            self.commits[repo_full_name].add(r['commit_sha'])


    def get_repos_with_at_least_n_commits(self, n):
        with_fix_count = []
        for x in self.commits:
            if len(self.commits[x]) >= n:
                with_fix_count.append((x, list(self.commits[x])))

        with_fix_count = sorted(with_fix_count, key=lambda x: len(x[1]), reverse=True)
        return with_fix_count

    def shas_for_repo(self, repo_full_name):
        if repo_full_name not in self.commits:
            return []

        return list(self.commits[repo_full_name])

    def is_security_related(self, repo_full_name, sha):
        if repo_full_name not in self.commits:
            return False
        
        if sha in self.commits[repo_full_name]:
            return True
        else:
            x = self.commits[repo_full_name]
            return False



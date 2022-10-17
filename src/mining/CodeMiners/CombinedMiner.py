from abc import abstractmethod
from pydriller import RepositoryMining, Commit
import pandas as pd
import json


class CombinedMiner:
    def __init__(self, miners_list):
        self.miners_list = miners_list


    def mine_repo(self, owner: str, repo: str, repo_path: str, commits: pd.DataFrame):
        self.commits = commits
        _samples = 0
        i = 0 
        for _, row in commits.iterrows():
            if i%100 == 0:
                print(owner, repo, i/commits.shape[0])
            i+=1
            try:
                for commit in RepositoryMining(repo_path, single=row['label_sha'], include_refs=True, include_remotes=True).traverse_commits():
                    for miner in self.miners_list:
                        miner.mine_commit2(owner, repo, commit, row['label_security_related'])

            except Exception as e:
                print('Finidng commit failed', owner, repo, row['label_sha'])
                print(e)

        print('DONE', owner, repo, _samples)
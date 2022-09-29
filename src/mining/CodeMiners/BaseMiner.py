from abc import abstractmethod
from typing import Iterable
from pydriller import RepositoryMining, Commit
from datetime import datetime
import json

start_year = 2010
checkpoint_frequency = 250

class BaseMiner:
    def __init__(self, checkpoints_directory):
        self.checkpoints_directory = checkpoints_directory

    @abstractmethod
    def mine_repo(self, owner: str, repo: str, repo_path: str, fix_commits: Iterable):
        pass

    @abstractmethod
    def _should_mine_commit(self, commit):
        pass


    @abstractmethod
    def _mine_commit(self, commit: Commit):
        pass


    def _iterate_commits(self, owner: str, repo: str, repo_path: str):
        data = {}
        for commit in RepositoryMining(repo_path, since=datetime(start_year, 1, 1)).traverse_commits():
            if self._should_mine_commit(commit):
                commit_data = self._mine_commit(owner, repo, commit)
                if commit_data != None and len(commit_data) > 0:
                    data[commit.hash] = commit_data

                if len(data) >= checkpoint_frequency:
                    self.save_to_checkpoint(data, owner, repo, commit.hash)
                    data = {}

        self.save_to_checkpoint(data, owner, repo, 'final')


    def save_to_checkpoint(self, data, owner, repo, hash):
        with open(f'{self.checkpoints_directory}/{owner}-{repo}-{hash}.json', 'w')  as f:
            json.dump(data, f)


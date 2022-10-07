import sys
sys.path.insert(0, r'D:\Projects\2022')


from pydriller import RepositoryMining,Commit
from os import path
import pandas as pd
import typing

from src.mining.CodeFeaturesMiners.CommitLevelFeatureMiner import CommitLevelFeatureMiner
from src.mining.CodeFeaturesMiners.FileLevelFeatureMiner import FileLevelFeatureMiner
from src.mining.CodeFeaturesMiners.AbstractMiner import AbstractMiner

import random
random.seed(42)


features_cache_location = 'results\checkpoints3'

class ExtraCommitMiner:
    def __init__(self, path_to_repository_store, log_fnc):
        self.file_miner = FileLevelFeatureMiner(AbstractMiner.features)
        self.commit_miner = CommitLevelFeatureMiner(AbstractMiner.features)
        self._repositories_location = path_to_repository_store
        self.log_fnc = log_fnc

        self.total_commits = 0
        self.commit_result = {}
        self.file_result = {}

    def mine_repos(self, repos):
        for repo in repos:
            self.mine_repo(repo)


    def mine_repo(self, repo_full_name, fix_commits=None, contributors=None):
        self.log_fnc('Processing repo', repo_full_name)

        self.commit_result[repo_full_name] = []
        self.file_result[repo_full_name] = []

        segments = repo_full_name.split('/')
        repo_path = path.join(self._repositories_location, segments[0], segments[1])

        pre_mined_commits = self._pre_mine_changes(repo_path)
        self.log_fnc(f'Done with premining {repo_full_name}')

        fix_commits_set =  set(fix_commits) if fix_commits else set()
        for index in range(len(pre_mined_commits)):
            if pre_mined_commits[index].hash in fix_commits_set:
                self._save_mine_commit(pre_mined_commits[index], index, pre_mined_commits, repo_full_name, contributors)
            
            if index - 200 >= 0:
                pre_mined_commits[index - 200] = None
        pre_mined_commits = None

        mined_in_frist_stage = set([x['label_sha'] for x in  self.commit_result[repo_full_name]])

        for x in fix_commits:
            if x in mined_in_frist_stage:
                continue
            try:
                for commit in RepositoryMining(repo_path, single=x).traverse_commits():
                    self._save_mine_commit(commit, None, None, repo_full_name, contributors)
                    print('Mined without history')
            except:
                self.log_fnc('Commit not found', repo_full_name, x)

        commit_df = pd.DataFrame(self.commit_result[repo_full_name])
        commit_df.to_csv(f'{features_cache_location}/commit-level-x-{segments[0]}-{segments[1]}.csv')
        file_df = pd.DataFrame(self.file_result[repo_full_name])
        file_df.to_csv(f'{features_cache_location}/file-level-x-{segments[0]}-{segments[1]}.csv')
        return commit_df, file_df

    def _pre_mine_changes(self, repo_path) -> typing.List[Commit]:
        pre_mined_commits = []
        for commit in RepositoryMining(repo_path, include_remotes=True, include_refs=True).traverse_commits():
            pre_mined_commits.append(commit)
        return pre_mined_commits


    def _save_mine_commit(self, commit, index, commits:typing.List[Commit], repo_full_name, contributors):
        try:
            self.mine_commit(commit, index, commits, repo_full_name, contributors)
        except Exception as e:
            self.log_fnc('CommitMiner', 'Failed to process commit data', index, repo_full_name)
            self.log_fnc(e)



    def mine_commit(self, commit, index, commits:typing.List[Commit], repo_full_name, contributors):
        commit_level_features = self.commit_miner.mine(repo_full_name, commit, commits, index, contributors)
        self.commit_result[repo_full_name].append(commit_level_features)

        for mod in commit.modifications:
            file_level_features = self.file_miner.mine_commit(repo_full_name, mod, commits, index)
            self.file_result[repo_full_name].append(file_level_features)

        self.total_commits += 1

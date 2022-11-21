import json
from pydriller import RepositoryMining,ModificationType,Commit,GitRepository
from datetime import datetime, date
import pytz
from os import path
import pandas as pd
import math
import typing

import random
random.seed(42)

from src.mining.CodeFeaturesMiners.CommitLevelFeatureMiner import CommitLevelFeatureMiner
from src.mining.CodeFeaturesMiners.FileLevelFeatureMiner import FileLevelFeatureMiner
from src.mining.CodeFeaturesMiners.AbstractMiner import AbstractMiner



START_YEAR = 2010
RATIO = 50


class CommitMiner:
    def __init__(self, path_to_repository_store, features_cache_location, log_fnc, ratio=RATIO):
        self.features_cache_location = features_cache_location
        self.file_miner = FileLevelFeatureMiner(AbstractMiner.features)
        self.commit_miner = CommitLevelFeatureMiner(AbstractMiner.features)
        self._repositories_location = path_to_repository_store
        self.log_fnc = log_fnc

        self.ratio = ratio
        self.total_commits = 0
        self.commit_result = {}
        self.file_result = {}



    def mine_repos(self, repos, start_year=START_YEAR):
        for repo in repos:
            self.mine_repo(repo, start_year)


    def mine_repo(self, repo_full_name, start_year=START_YEAR, fix_commits=None, contributors=None):
        self.log_fnc('Processing repo', repo_full_name)

        self.commit_result[repo_full_name] = []
        self.file_result[repo_full_name] = []

        segments = repo_full_name.split('/')
        repo_path = path.join(self._repositories_location, segments[0], segments[1])

        pre_mined_commits = self._pre_mine_changes(repo_path, start_year)
        self.log_fnc(f'Done with premining {repo_full_name}')

        if len(pre_mined_commits) == 0:
            self.log_fnc(f'NO COMMITS ON MAIN BRANCH', repo_full_name)


        if fix_commits != None:
            probability_of_mine = len(fix_commits)*self.ratio / len(pre_mined_commits)
            self.log_fnc(f'Mining {len(fix_commits)} fix commits and some commits with probability {probability_of_mine}')

        fix_commits_set =  set(fix_commits) if fix_commits else set()
        for index in range(len(pre_mined_commits)):
            if random.random() <= probability_of_mine:
                self._save_mine_commit(index, pre_mined_commits, repo_full_name, contributors)
            elif pre_mined_commits[index].hash in fix_commits_set:
                self._save_mine_commit(index, pre_mined_commits, repo_full_name, contributors)
            
            if index - 200 >= 0:
                pre_mined_commits[index - 200] = None

        commit_df = pd.DataFrame(self.commit_result[repo_full_name])
        commit_df.to_csv(f'{self.features_cache_location}/commit-level-{segments[0]}-{segments[1]}.csv')
        file_df = pd.DataFrame(self.file_result[repo_full_name])
        file_df.to_csv(f'{self.features_cache_location}/file-level-{segments[0]}-{segments[1]}.csv')
        return commit_df, file_df


    def _pre_mine_changes(self, repo_path, start_year) -> typing.List[Commit]:
        pre_mined_commits = []
        for commit in RepositoryMining(repo_path, since=datetime(start_year,1,1), include_refs=True, include_remotes=True).traverse_commits():
            pre_mined_commits.append(commit)
        return pre_mined_commits


    def _save_mine_commit(self, index, commits:typing.List[Commit], repo_full_name, contributors):
        try:
            self.mine_commit(index, commits, repo_full_name, contributors)
        except:
            self.log_fnc('CommitMiner', 'Failed to process commit data', index, repo_full_name)



    def mine_commit(self, index, commits:typing.List[Commit], repo_full_name, contributors):
        commit = commits[index]
        
        commit_level_features = self.commit_miner.mine(repo_full_name, commit, commits, index, contributors)
        self.commit_result[repo_full_name].append(commit_level_features)

        for mod in commit.modifications:
            file_level_features = self.file_miner.mine_commit(repo_full_name, mod, commits, index)
            self.file_result[repo_full_name].append(file_level_features)

        self.total_commits += 1

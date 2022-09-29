import json
from pydriller import RepositoryMining,ModificationType,Commit,GitRepository
from datetime import datetime, date
import pytz
from os import path
import pandas as pd
import math

import sys
sys.path.insert(0, r'D:\Projects\2022')

from src.linked_commit_investigation.CommitLevelFeatureMiner import CommitLevelFeatureMiner
from src.linked_commit_investigation.FileLevelFeatureMiner import FileLevelFeatureMiner
from src.linked_commit_investigation.AbstractMiner import AbstractMiner



import random
random.seed(42)

from src.linked_commit_investigation.RepoDownloader import RepoDownloader

# START_YEAR = 2018
START_YEAR = 2010
# LIMIT = 50 
LIMIT = None
RATIO = 49
features_cache_location = 'results\checkpoints1'
all_repos_path = '/d/Projects/data/repos'
premined_checkpoint = r'D:\Projects\2022\results\premined'

class CommitMiner:
    def __init__(self, path_to_repository_store, logging = True):
        self.file_miner = FileLevelFeatureMiner(AbstractMiner.features)
        self.commit_miner = CommitLevelFeatureMiner(AbstractMiner.features)
        self.RepoDownloader = RepoDownloader(path_to_repository_store)
        self._repositories_location = path_to_repository_store
        self._logging=logging

        self.total_commits = 0
        self.commit_result = {}
        self.file_result = {}



    def mine_repos(self, repos, start_year=START_YEAR):
        for repo in repos:
            self.RepoDownloader.download_repos([repo])
            self.mine_repo_2(repo, start_year)


    def mine_repo_2(self, repo_full_name, start_year=START_YEAR, fix_commits=None, partial_mine = False):
        if partial_mine and fix_commits==None:
            raise Exception('Invalid setup')
        if self._logging:
            print('Processing repo', repo_full_name)

        self.commit_result[repo_full_name] = []
        self.file_result[repo_full_name] = []

        segments = repo_full_name.split('/')
        repo_path = path.join(self._repositories_location, segments[0], segments[1])
        gr = GitRepository(repo_path)
        all_commit_count = gr.total_commits()
        if fix_commits != None:
            probability_of_mine = len(fix_commits)*RATIO / all_commit_count
            if self._logging:
                print(f'Mining {len(fix_commits)} fix commits and some commits with probability {probability_of_mine}')

        pre_mined_commits = [None] * all_commit_count
        index = -1
        for commit in RepositoryMining(repo_path, since=datetime(start_year,1,1)).traverse_commits():
            index += 1
            pre_mined_commits[index] = {
                'author_date': commit.author_date,
                'merge': commit.merge,
            }
        
        pre_mined_commits = [x for x in pre_mined_commits if x]
        dfx = pd.DataFrame(pre_mined_commits)
        dfx.to_csv(f'premined_checkpoint-{segments[0]}-{segments[1]}.csv', index=False)

        if self._logging:
            print(f'Done with premining {repo_full_name}')

        fix_commits_set =  set([x['sha'] for x in fix_commits]) if fix_commits else set()
        index = -1
        for commit in RepositoryMining(repo_path, since=datetime(start_year,1,1)).traverse_commits():
            index += 1
            if LIMIT != None and len(self.commit_result[repo_full_name]) >= LIMIT:
                break
            
            history_before = pre_mined_commits[max(0, index-20):]
            history_after = pre_mined_commits[index:min(all_commit_count-1, index+10)]
            
            if not partial_mine:
                self._save_mine(commit, repo_full_name, history_before, history_after)
            else:
                if random.random() <= probability_of_mine:
                    self._save_mine(commit, repo_full_name, history_before, history_after)
                elif commit.hash[:10] in fix_commits_set:
                    self._save_mine(commit, repo_full_name, history_before, history_after)

        commit_df = pd.DataFrame(self.commit_result[repo_full_name])
        commit_df.to_csv(f'{features_cache_location}/commit-level-{segments[0]}-{segments[1]}.csv')
        file_df = pd.DataFrame(self.file_result[repo_full_name])
        file_df.to_csv(f'{features_cache_location}/file-level-{segments[0]}-{segments[1]}.csv')


    def _save_mine(self, commit, repo_full_name, history_before, history_after):
        try:
            self.mine_commit(commit, repo_full_name, history_before, history_after)
            print('CommitMiner', f'Processed {self.total_commits} commit', commit.hash, repo_full_name)
        except:
            print('CommitMiner', 'Failed to process commit data', commit.hash, repo_full_name)



    def mine_commit(self, commit:Commit, repo_full_name, history_before, history_after):
        commit_level_features = self.commit_miner.mine(commit, history_before, history_after)
        commit_level_features['label_repo_full_name']= repo_full_name
        self.commit_result[repo_full_name].append(commit_level_features)
        for mod in commit.modifications:
            file_level_features = self.file_miner.mine_commit(mod)
            file_level_features['label_repo_full_name'] = repo_full_name
            file_level_features['label_sha']= commit.hash
            file_level_features['label_commit_date']= commit.committer_date
            self.file_result[repo_full_name].append(file_level_features)
        self.total_commits += 1




def validate_on_selected_repos():
    cm = CommitMiner(all_repos_path)
    npm_repos = [ 'moment/moment','lodash/lodash','eslint/eslint','webpack/webpack-dev-server','caolan/async']
    pypi_repos = ['psf/requests', 'numpy/numpy','django/django', 'python-pillow/Pillow','scipy/scipy']
    mvn_repos = ['junit-team/junit4','google/guava','spring-projects/spring-framework','h2database/h2database','google/gson']
    repos = npm_repos + pypi_repos + mvn_repos
    repos = ['moment/moment']

    cm.mine_repos(repos)


if __name__ == '__main__':
    validate_on_selected_repos()
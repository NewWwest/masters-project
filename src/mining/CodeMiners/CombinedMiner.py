from datetime import datetime
from pydriller import RepositoryMining, Commit, GitRepository
import pandas as pd
import json

import random
random.seed(42)

from src.utils.utils import info_log, warn_log

START_YEAR = 2017
RATIO = 50

class CombinedMiner:
    def __init__(self, miners_list, result_uploader = None, zipped_results_directory = '', ratio = RATIO):
        self.miners_list = list(miners_list)
        self.result_uploader = result_uploader
        self.zipped_results_directory = zipped_results_directory
        self.ratio = ratio


    # For mining specific commits and random background ones
    def mine_with_background(self, owner: str, repo: str, repo_path: str, commits: set, start_year = START_YEAR):
        found_commits = 0
        mined_files = []
        for cm in commits:
            try:
                for commit in RepositoryMining(repo_path, single=cm).traverse_commits():
                    found_commits += 1
                    self._iterate_miners(owner, repo, mined_files, commit, True)
            except Exception as e:
                info_log('Finidng commit failed', owner, repo, cm)
                info_log(e)
                
        info_log('MINED Positive', owner, repo, len(commits), found_commits)
        if found_commits == 0:
            return

        repo_info = GitRepository(repo_path)
        total_commits = repo_info.total_commits()
        probability_of_mine = found_commits*self.ratio / total_commits

        for commit in RepositoryMining(repo_path, since=datetime(start_year,1,1), include_remotes=True).traverse_commits():
            if random.random() <= probability_of_mine and commit.hash not in commits:
                found_commits+=1
                self._iterate_miners(owner, repo, mined_files, commit, False)
                        
        info_log('MINED Background', owner, repo, len(commits), found_commits)

        if self.result_uploader != None:
            self.result_uploader.upload_files_as_zip(mined_files, f'{owner}-{repo}', self.zipped_results_directory)


    def _iterate_miners(self, owner, repo, mined_files, commit, sec_related):
        for miner in self.miners_list:
            try:
                filenames = miner._mine_commit(owner, repo, commit, sec_related)
                if filenames != None and len(filenames)>0:
                    mined_files+=filenames
            except Exception as e:
                warn_log('Processing commit failed', owner, repo, commit.hash)
                warn_log(miner.__class__.__name__)
                warn_log(e)


    # For mining specific commits
    def mine_repo(self, owner: str, repo: str, repo_path: str, commits: pd.DataFrame):
        if commits.shape[0] == 0:
            return

        found_commits =0
        mined_files = []
        for _, row in commits.iterrows():
            try:
                for commit in RepositoryMining(repo_path, single=row['label_sha']).traverse_commits():
                    found_commits += 1
                    self._iterate_miners(owner, repo, mined_files, commit, row['label_security_related'])
            except Exception as e:
                info_log('Finidng commit failed', owner, repo, row['label_sha'])
                info_log(e)
                
        info_log('DONE', owner, repo, commits.shape[0], found_commits, len(mined_files))
        info_log('DONE')

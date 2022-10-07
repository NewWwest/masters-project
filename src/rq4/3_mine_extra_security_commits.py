import sys
sys.path.insert(0, r'D:\Projects\aaa')

from src.rq4.CommitProvider import CommitProvider
from src.proxies.RepoDownloader import RepoDownloader
from src.mining.CodeFeaturesMiners.ExtraCommitMiner import ExtraCommitMiner
from src.mining.CodeFeaturesMiners.FeatureCalculator import FeatureCalculator

from src.utils.utils import get_files_in_from_directory

import os
import random
random.seed(42)
import time
import json
import tqdm
import pandas as pd
from multiprocessing import Pool
import logging

security_commits_file = 'src/datasets/security_related_commits_in_vuln.csv'
contributors_file = 'src/datasets/contributors.json'

results_location = 'results\checkpoints1'
fix_location = 'results\checkpoints3'

# data_files = get_files_in_from_directory(features_location, '.csv', startswith= 'features')
# dfs = [pd.read_csv(f) for f in data_files]
# df = pd.concat(dfs)
# commitProvider = CommitProvider(security_commits_file)

# df['label_security_related'] = df.apply(lambda r: commitProvider.is_security_related(r['label_repo_full_name'], r['label_sha']), axis=1)

cpus = 3

repositories_path = '/rr'

def logx(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.warning(log_message)

def mine_repo(repo_fix_commits):
    try:
        repo_full_name = repo_fix_commits[0]
        segments = repo_full_name.split('/')

        repo_alreaddy_fixed = os.path.exists(f'{fix_location}/features-{segments[0]}-{segments[1]}.csv')
        if repo_alreaddy_fixed:
            # logx('Repo alreaddy processed with fix script', repo_full_name)
            return

        repo_alreaddy_done = os.path.exists(f'{results_location}/features-{segments[0]}-{segments[1]}.csv')
        if repo_alreaddy_done:
            temp = pd.read_csv(f'{results_location}/features-{segments[0]}-{segments[1]}.csv')
            mined_in_first_run = set(temp['label_sha'])
        else:
            mined_in_first_run = set()

        need_to_be_mined_extra = set(repo_fix_commits[1]).difference(mined_in_first_run)
        if len(need_to_be_mined_extra) == 0:
            return


        with open(contributors_file, 'r') as f:
            contributors = json.load(f)

        rd = RepoDownloader(repositories_path)
        rd.download_repos([repo_full_name])
        cm = ExtraCommitMiner(repositories_path, logx)
        contributors = contributors[repo_full_name] if repo_full_name in contributors else None
        commit_level, file_level = cm.mine_repo(repo_full_name, fix_commits=need_to_be_mined_extra, contributors=contributors)
        fc = FeatureCalculator()
        fc.process_repo(commit_level, file_level, repo_full_name)
        # rd.remove_repos([repo_full_name])
    except Exception as e:
        logx('FAIL', repo_full_name)
        # logx(e)


def main():
    commitProvider = CommitProvider(security_commits_file)
    repos = commitProvider.get_repos_with_at_least_n_commits(5)

    # for r in repos:
    #     mine_repo(r)

    with Pool(cpus) as p:
        with tqdm.tqdm(total=len(repos)) as pbar:
            for _ in p.imap_unordered(mine_repo, repos, chunksize=1):
                pbar.update()



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

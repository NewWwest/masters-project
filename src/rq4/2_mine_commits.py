import sys
sys.path.insert(0, r'D:\Projects\aaa')

from src.rq4.CommitProvider import CommitProvider
from src.proxies.RepoDownloader import RepoDownloader
from src.mining.CodeFeaturesMiners.CommitMiner import CommitMiner
from src.mining.CodeFeaturesMiners.FeatureCalculator import FeatureCalculator

import os
import random
random.seed(42)
import time
import json
import tqdm
from multiprocessing import Pool
import logging

security_commits_file = 'src/datasets/security_related_commits_in_vuln.csv'
contributors_file = 'src/datasets/contributors.json'
results_location = 'results/checkpoints1'
cpus = 1

repositories_path = '/rr3'

def logx(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.warning(log_message)

def mine_repo(repo_fix_commits):
    try:
        repo_full_name = repo_fix_commits[0]
        segments = repo_full_name.split('/')

        repo_alreaddy_done = os.path.exists(f'{results_location}/features-{segments[0]}-{segments[1]}.csv')
        if repo_alreaddy_done:
            # logx('repo alreaddy done', repo_full_name)
            return

        with open(contributors_file, 'r') as f:
            contributors = json.load(f)

        rd = RepoDownloader(repositories_path)
        rd.download_repos([repo_full_name])
        cm = CommitMiner(repositories_path, logx)
        contributors = contributors[repo_full_name] if repo_full_name in contributors else None
        commit_level, file_level = cm.mine_repo(repo_full_name, fix_commits=repo_fix_commits[1], contributors=contributors)
        fc = FeatureCalculator()
        fc.process_repo(commit_level, file_level, repo_full_name)
        rd.remove_repos([repo_full_name])
    except Exception as e:
        logx('Something went wrong')
        logx('Something went wrong')
        logx(e)


def main():
    commitProvider = CommitProvider(security_commits_file)
    repos = commitProvider.get_repos_with_at_least_n_commits(5)

    for r in repos:
        mine_repo(r)

    # with Pool(cpus) as p:
    #     with tqdm.tqdm(total=len(repos)) as pbar:
    #         for _ in p.imap_unordered(mine_repo, repos, chunksize=1):
    #             pbar.update()



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

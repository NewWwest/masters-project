#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

from src.proxies.RepoDownloader import RepoDownloader
from src.mining.CodeFeaturesMiners.CommitMiner import CommitMiner
from src.mining.CodeFeaturesMiners.FeatureCalculator import FeatureCalculator
from src.utils.utils import get_files_in_from_directory, info_log, warn_log

import pandas as pd
import os
import random
random.seed(42)
import time
import json
import tqdm
from multiprocessing import Pool

# CSV of security related commits
path_to_security_commits = 'data/security_related_commits_in_vuln.csv'
# JSON of top contributors of the repository
path_to_contributors_file = 'data/contributors.json'

# path to where the repositories will be downloaded to
repositories_path = '/rr3'
# path to where the features from each repository will be saved to
features_location = 'results/features1'
# path to where the commit and file level intermediate results will be saved to
file_and_commit_level_metrics_location = 'results\features1'

# the number of concurrent jobs to use (with multiprocessing.Pool)
cpus = 20


def mine_repo(repo_fix_commits):
    try:
        repo_full_name = repo_fix_commits[0]
        segments = repo_full_name.split('/')

        repo_alreaddy_done = os.path.exists(f'{features_location}/features-{segments[0]}-{segments[1]}.csv')
        if repo_alreaddy_done:
            info_log('repo alreaddy done...', repo_full_name)
            return

        info_log('Processing...', repo_full_name)
        with open(path_to_contributors_file, 'r') as f:
            contributors = json.load(f)

        rd = RepoDownloader(repositories_path)
        rd.download_repos([repo_full_name])
        cm = CommitMiner(repositories_path, file_and_commit_level_metrics_location, info_log)
        contributors = contributors[repo_full_name] if repo_full_name in contributors else None
        commit_level, file_level = cm.mine_repo(repo_full_name, fix_commits=repo_fix_commits[1], contributors=contributors)
        fc = FeatureCalculator(features_location)
        fc.process_repo(commit_level, file_level, repo_full_name)
        rd.remove_repos([repo_full_name])
    except Exception as e:
        warn_log('Something went wrong')
        warn_log(e)


def main():
    security_commits = pd.read_csv(path_to_security_commits)
    security_commits['label_repo_full_name'] = security_commits.apply(lambda r: f'{r["repo_owner"]}/{r["repo_name"]}', axis=1)
    by_repo = security_commits.groupby('label_repo_full_name')
    input = [(r, set(cs['commit_sha'])) for r, cs in by_repo]

    if cpus == 1:
        for r in input:
            mine_repo(r)
    else:
        with Pool(cpus) as p:
            with tqdm.tqdm(total=len(input)) as pbar:
                for _ in p.imap_unordered(mine_repo, input, chunksize=1):
                    pbar.update()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

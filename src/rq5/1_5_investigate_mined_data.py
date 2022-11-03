import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import json
import time
from multiprocessing import Pool
from os import path
from src.proxies.RepoDownloader import RepoDownloader

import random
random.seed(42)
import numpy as np
np.random.seed(42) 

from src.utils.utils import get_files_in_from_directory 

from src.mining.CodeMiners.CombinedMiner import CombinedMiner 
from src.mining.CodeMiners.RollingWindowMiner import RollingWindowMiner 
from src.mining.CodeMiners.AddedCodeMiner import AddedCodeMiner 
from src.mining.CodeMiners.CodeParserMiner import CodeParserMiner 
from src.mining.CodeMiners.SampleEncoder import SampleEncoder 


cpus = 10
repositories_path = '/repolist2'
path_to_security_commits = r'D:\Projects\aaa\results\rq4_results\features.csv'

results_location_RollingWindowMiner = 'results/dl/RollingWindowMiner'
results_location_AddedCodeMiner = 'results/dl/AddedCodeMiner'
results_location_AST =  'results/dl/CodeParserMiner_ast'
results_location_Actions =  'results/dl/CodeParserMiner_edit'

allx = r'D:\Projects\aaa\results\rq4_results\features.csv'
npm = r'D:\Projects\aaa\results\rq4_results\features_npm.csv'
pypi = r'D:\Projects\aaa\results\rq4_results\features_pypi.csv'
mvn = r'D:\Projects\aaa\results\rq4_results\features_mvn.csv'

df_all = pd.read_csv(allx)




def main():
    commits_df = pd.read_csv(path_to_security_commits)
    by_repo = commits_df.groupby('label_repo_full_name')
    processed_comits, processed_repos = get_processed_commits()
    mined = 0
    not_mined = 0
    for r, df in by_repo:
        for _, row in df.iterrows():
            if row['label_sha'] in processed_comits:
                mined += 1
            else:
                not_mined += 1

    print(mined/not_mined)
    


def get_processed_commits():
    result_dirs = [
        results_location_RollingWindowMiner,
        results_location_AddedCodeMiner,
        results_location_AST,
        results_location_Actions,
    ]
    processed_comits = set()
    processed_repos = set()
    for rd in result_dirs:
        for f in get_files_in_from_directory(rd):
            filename = path.splitext(f)[0]
            segments = filename.split('-')
            processed_comits.add(segments[-1])
            processed_repos.add(f'{segments[2]}/{segments[3]}')

    return processed_comits, processed_repos


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
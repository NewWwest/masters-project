import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import time
from multiprocessing import Pool
from os import path
from src.proxies.RepoDownloader import RepoDownloader

import random
random.seed(42)
import numpy as np
np.random.seed(42) 

from src.mining.CodeMiners.CombinedMiner import CombinedMiner 
from src.mining.CodeMiners.RollingWindowMiner import RollingWindowMiner 
from src.mining.CodeMiners.AddedCodeMiner import AddedCodeMiner 
from src.mining.CodeMiners.CodeParserMiner import CodeParserMiner 
from src.mining.CodeMiners.SampleEncoder import SampleEncoder 


cpus = 10
repositories_path = '/repolist2'
path_to_security_commits = r'D:\Projects\aaa\results\rq4_results\features.csv'
results_location_RollingWindowMiner = 'results/new_mined_code_rolling_window'
results_location_AddedCodeMiner = 'results/new_mined_code_added_code'
results_location_AST =  'results/new_mined_code_ast_code'
results_location_Actions =  'results/new_mined_code_actions_code'


base_model = 'microsoft/graphcodebert-base'
valid_extensions = set()
npm_code  = ['js', 'jsx', 'ts', 'tsx', ]
npm_like_code  = ['cjs', 'mjs', 'iced', 'liticed', 'coffee', 'litcoffee', 'ls', 'es6', 'es', 'sjs', 'eg']
java_code  = ['java']
java_like_code  = ['jnl', 'jar', 'class', 'dpj', 'jsp', 'scala', 'sc', 'kt', 'kts', 'ktm']
pypi_code = ['py', 'py3']
pypi_code_like = ['pyw', 'pyx', 'ipynb']
valid_extensions.update(npm_code)
valid_extensions.update(npm_like_code)
valid_extensions.update(java_code)
valid_extensions.update(java_like_code)
valid_extensions.update(pypi_code)
valid_extensions.update(pypi_code_like)


def main():
    commits_df = pd.read_csv(path_to_security_commits)
    by_repo = commits_df.groupby('label_repo_full_name')

    # for x in by_repo:
    #     mine_a_repo(x)

    with Pool(cpus) as p:
        p.map(mine_a_repo, by_repo, chunksize=1)


def mine_a_repo(data):
    repo_full_name = data[0]
    segments = repo_full_name.split('/')
    commits_df = data[1]

    try:
        print("MINING", repo_full_name)
        try:
            rd = RepoDownloader(repositories_path)
            repo_paths = rd.download_repos([repo_full_name])
        except Exception as e:
            print('Downloading repo failed but attempting to process anyway', repo_full_name)
            print(e)
            repo_paths = [path.join(repositories_path, segments[0], segments[1])]

        sample_encodder = SampleEncoder(base_model, valid_extensions)
        
        miner1 = RollingWindowMiner(results_location_RollingWindowMiner, sample_encodder, valid_extensions)
        miner2 = AddedCodeMiner(results_location_AddedCodeMiner, sample_encodder, valid_extensions)
        # miner3 = CodeParserMiner(results_location_AddedCodeMiner, sample_encodder, valid_extensions)
        
        combined_miner = CombinedMiner([miner1, miner2])

        combined_miner.mine_repo(segments[0], segments[1], repo_paths[0], commits_df)

        rd.remove_repos([repo_full_name])
        print("DONE", repo_full_name)
    except Exception as e:
        print()
        print('Failed to process repo', repo_full_name)
        print(e)
        print()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
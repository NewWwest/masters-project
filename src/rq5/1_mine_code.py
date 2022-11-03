import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import json
import time
from multiprocessing import Pool
from os import path
from src.proxies.RepoDownloader import RepoDownloader
from src.utils.utils import get_files_in_from_directory 

from src.mining.CodeMiners.CombinedMiner import CombinedMiner 
from src.mining.CodeMiners.RollingWindowMiner import RollingWindowMiner 
from src.mining.CodeMiners.AddedCodeMiner import AddedCodeMiner 
from src.mining.CodeMiners.CodeParserMiner import CodeParserMiner 
from src.mining.CodeMiners.SampleEncoder import SampleEncoder 


cpus = 10
repositories_path = '/repolist2'
path_to_security_commits = r'D:\Projects\aaa\results\rq4_results\features.csv'
input_data_location = 'results/checkpoints_fixMapper'

results_location_RollingWindowMiner = 'results/dl/RollingWindowMiner'
results_location_AddedCodeMiner = 'results/dl/AddedCodeMiner'
results_location_AST =  'results/dl/CodeParserMiner_ast'
results_location_Actions =  'results/dl/CodeParserMiner_edit'


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

_extensions_to_ignore = ['md', 'json' , 'txt', 'gradle', 'sha', 'lock', 'ruby-version', 'yaml', 'yml', 'xml', 'html', 'gitignore']
extensions_to_ignore = set(_extensions_to_ignore)

def main():
    commits_df = pd.read_csv(path_to_security_commits)
    processed_comits, somewhat_processed_repos = get_processed_commits()

    commits_df_filtered = commits_df[commits_df.apply(lambda row: row['label_sha'] not in processed_comits, axis=1)]

    by_repo = commits_df_filtered.groupby('label_repo_full_name')

    by_repo = sorted(by_repo, key=lambda x: -x[1].shape[0])
    # for x in by_repo:
    #     print(x[0])
    
    with Pool(cpus) as p:
        p.map(mine_a_repo, by_repo, chunksize=1)

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
            processed_repos
            processed_comits.add(segments[-1])
            processed_repos.add(f'{segments[2]}/{segments[3]}')

    return processed_comits, processed_repos



def mine_a_repo(data):
    repo_full_name = data[0]
    segments = repo_full_name.split('/')
    commits_df = data[1]

    try:
    # if True:
        print("MINING", repo_full_name)
        try:
            rd = RepoDownloader(repositories_path)
            repo_paths = rd.download_repos([repo_full_name])
        except Exception as e:
            print('Downloading repo failed but attempting to process anyway', repo_full_name)
            print(e)
            repo_paths = [path.join(repositories_path, segments[0], segments[1])]

        sample_encodder = SampleEncoder(base_model)
        
        miner1 = RollingWindowMiner(results_location_RollingWindowMiner, sample_encodder, 
            valid_extensions=None, extensions_to_ignore=extensions_to_ignore)
        miner2 = AddedCodeMiner(results_location_AddedCodeMiner, sample_encodder, 
            valid_extensions=None, extensions_to_ignore=extensions_to_ignore)
        miner3 = CodeParserMiner(results_location_AST, results_location_Actions, sample_encodder, 
            valid_extensions=None, extensions_to_ignore=extensions_to_ignore)
        
        combined_miner = CombinedMiner([miner1, miner2, miner3])

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
    
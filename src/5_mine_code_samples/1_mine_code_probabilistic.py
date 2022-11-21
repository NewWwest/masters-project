#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import pandas as pd
import time
from multiprocessing import Pool
import os
from os import path
import random 


# from src.proxies.GcCloudStorageProxy import GcCloudStorageProxy as UploaderProxy
from src.proxies.LocalStorageProxy import LocalStorageProxy as UploaderProxy

from src.proxies.RepoDownloader import RepoDownloader
from src.utils.utils import get_files_in_from_directory, info_log, warn_log
from src.mining.CodeMiners.CombinedMiner import CombinedMiner 
from src.mining.CodeMiners.RollingWindowMiner import RollingWindowMiner 
from src.mining.CodeMiners.AddedCodeMiner import AddedCodeMiner 
from src.mining.CodeMiners.CodeParserMiner import CodeParserMiner 
from src.mining.CodeMiners.SampleEncoder import SampleEncoder 
from src.mining.CodeMiners.VulFixMinerMiner import VulFixMinerMiner 
from src.mining.CodeMiners.CommitSizeMiner import CommitSizeMiner 

# Input files of security commits to mine
path_to_security_commits = 'data/security_relevant_commits.csv'


random.seed(42)
cpus = 10
# Path where repositories will be downloaded to
repositories_path = '/rr'
# If using the CodeParserMiner one needs to run the CodeParser java project and configure the variables in GumTreeProxy
# GumTreeProxy.parser_url = 'http://localhost:8000'
# GumTreeProxy.workdir = 'workdir'


# The result dierectories where individual mined commit are saved to 
# If one runs the tool with the UploaderProxy individual commits are zippe together into one zip file per repository
results_location_RollingWindowMiner = 'results/dl3/RollingWindowMiner'
results_location_AddedCodeMiner = 'results/dl3/AddedCodeMiner'
results_location_AST =  'results/dl3/CodeParserMiner_ast'
results_location_Actions =  'results/dl3/CodeParserMiner_edit'
results_location_VulFixMiner =  'results/dl3/VulFixMiner'
results_location_CommitSizeMiner =  'results/dl3/CommitSizeMiner'
zipped_results_directory =  'results/zipped'
directories_to_make = [
    results_location_RollingWindowMiner,
    results_location_AddedCodeMiner,
    results_location_AST,
    results_location_Actions,
    results_location_VulFixMiner,
    results_location_CommitSizeMiner,
    zipped_results_directory
]


# The extensions to consider during the mining process
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

# The extensions to ignore during the mining process
_extensions_to_ignore = ['md', 'json' , 'txt', 'gradle', 'sha', 'lock', 'ruby-version', 'yaml', 'yml', 'xml', 'html', 'gitignore']
extensions_to_ignore = set(_extensions_to_ignore)


# The base model of the model for which the samples are prepared for - used to generate encodings with AutoTokenizer
base_model = 'microsoft/graphcodebert-base'

def make_dirs():
    for x in directories_to_make:
        try:
            os.makedirs(x)
        except:
            pass
        

def main():
    commits_df = pd.read_csv(path_to_security_commits)
    commits_df['repo_full_name'] = commits_df.apply(lambda x:f'{x["repo_owner"]}/{x["repo_name"]}', axis=1)
    by_repo = commits_df.groupby('repo_full_name')

    # by_repo = random.sample(list(by_repo), 5)
    # by_repo = sorted(by_repo, key=lambda x: -x[1].shape[0])

    if cpus == 1:
        for x in by_repo:
            mine_a_repo(x)
    else:
        with Pool(cpus) as p:
            p.map(mine_a_repo, by_repo, chunksize=1)


def mine_a_repo(data):
    info_log("MINING", repo_full_name)
    repo_full_name = data[0]
    segments = repo_full_name.split('/')
    commits_df = data[1]

    try:
        uploader = UploaderProxy()
        if uploader.check_exists(f'{segments[0]}-{segments[1]}', zipped_results_directory):
            info_log("SKIPPING", repo_full_name)
            return 

        try:
            rd = RepoDownloader(repositories_path)
            repo_paths = rd.download_repos([repo_full_name])
        except Exception as e:
            warn_log('Downloading repo failed but attempting to process anyway', repo_full_name)
            warn_log(e)
            repo_paths = [path.join(repositories_path, segments[0], segments[1])]

        sample_encodder = SampleEncoder(base_model)
        
        miner1 = RollingWindowMiner(results_location_RollingWindowMiner, sample_encodder, 
            rolling_line_window = 10,
            max_samples_per_commit = 100,
            valid_extensions=valid_extensions, 
            extensions_to_ignore=extensions_to_ignore)
        miner2 = AddedCodeMiner(results_location_AddedCodeMiner, sample_encodder, 
            valid_extensions=valid_extensions, 
            extensions_to_ignore=extensions_to_ignore)
        miner3 = CodeParserMiner(results_location_AST, results_location_Actions, sample_encodder, 
            max_samples_per_commit = 100,
            valid_extensions=valid_extensions, 
            extensions_to_ignore=extensions_to_ignore)
        miner4 = VulFixMinerMiner(
            results_dir=results_location_VulFixMiner,
            valid_extensions=valid_extensions, 
            extensions_to_ignore=extensions_to_ignore)
        miner5 = CommitSizeMiner(
            results_dir = results_location_CommitSizeMiner,
            extensions_to_ignore = extensions_to_ignore
        )

        combined_miner = CombinedMiner([miner1, miner2, miner3, miner4, miner5], uploader, zipped_results_directory)

        sec_commits = set(commits_df['commit_sha'])
        combined_miner.mine_with_background(segments[0], segments[1], repo_paths[0], sec_commits)

        rd.remove_repos([repo_full_name])
        info_log("DONE", repo_full_name)
    except Exception as e:
        warn_log('Failed to process repo', repo_full_name)
        warn_log(e)


if __name__ == '__main__':
    make_dirs()
    start_time = time.time()
    main()
    info_log("--- %s seconds ---" % (time.time() - start_time))
    
from multiprocessing import Pool
import time
import json
from pydriller import RepositoryMining, GitRepository
from datetime import datetime
import regex as re
import json
import shutil
from git import Repo
from os import path
from google.cloud import storage
import logging

start_year = 2017 
checkpoint_frequency = 1000

def logx(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.warning(log_message)


class KeywordFinder:
    def __init__(self, keywords, checkpoints_directory, bucket_proxy):
        self.checkpoints_directory = checkpoints_directory
        self.bucket_proxy = bucket_proxy
        self.keywords = {k: re.compile(r'(?i)\b' + k + r'\b') for k in keywords}

    def mine_repo(self, owner: str, repo: str, repo_path: str):
        processed_commits = 0
        result = []
        for commit in RepositoryMining(repo_path, since=datetime(start_year, 1, 1)).traverse_commits():
            processed_commits += 1
            for changeFile in commit.modifications:
                for k in self.keywords:
                    if self.keywords[k].search(changeFile.diff):
                        res = {
                            'owner':owner,
                            'repo':repo,
                            'hash':commit.hash,
                            'path':changeFile.new_path if changeFile.new_path != None else changeFile.old_path,
                            'keyword': k
                        }
                        result.append(res)


            if len(result) >= checkpoint_frequency:
                self.save_to_checkpoint(result, owner, repo, commit.hash, processed_commits)
                processed_commits = 0
                result = []

        self.save_to_checkpoint(result, owner, repo, 'final', processed_commits)
        processed_commits = 0
        result = []


    def save_to_checkpoint(self, data, owner, repo, hash, processed_commits):
        temp_data = {
            'processed_commits':processed_commits,
            'data': data
        }
        self.bucket_proxy.create_file(f'keywords_in_diff-{owner}-{repo}-{hash}.json', temp_data)


github_home = 'https://github.com'
class RepoDownloader:
    def __init__(self, path_to_store):
        self._path_to_store = path_to_store

    def download_repos(self, repo_list, force=False):
        paths = []
        for repo_full_name in repo_list:
            segments = repo_full_name.split('/')
            repo_path = path.join(self._path_to_store, segments[0], segments[1])
            paths.append(repo_path)
            repo_link = f'{github_home}/{repo_full_name}'

            if(not path.exists(repo_path)):
                logx('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)
            elif force:
                self.remove_repos([repo_full_name])
                logx('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)

        return paths

    def remove_repos(self, repo_list):
        for repo_full_name in repo_list:
            segments = repo_full_name.split('/')
            repo_path = path.join(self._path_to_store, segments[0], segments[1])
            if(path.exists(repo_path)):
                logx('RepoDownloader', 'Removing repo', repo_path)
                try:    
                    # watch out for PermissionError: [WinError 5] Access is denied
                    shutil.rmtree(repo_path, ignore_errors=True)
                except Exception as e:
                    logging.exception(e)


class BucketProxy:
    def __init__(self) -> None:
        self.storage_client = storage.Client()
        self.bucket_name = 'clean-equinox-keyword-test'
        pass

    def create_file(self, destination_blob_name, data):
        destination_blob_name = 'results/' + destination_blob_name
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(json.dumps(data))
        logx('Saved file', destination_blob_name )


repositories_path = '/home/awestfal/repos'
result_location = r'results\keywords_find'
repositories_to_process = 'repos_to_process.txt'
keywords_to_look_for = 'keywords.txt'

def main():
    bucket_proxy = BucketProxy()
    with open(repositories_to_process, 'r') as f:
        repos = f.readlines()
    repos = list([r.strip() for r in repos])
    for x in repos:
        mine_a_repo(x, bucket_proxy)


def mine_a_repo(repo_full_name, bucket_proxy):
    segments = repo_full_name.split('/')

    try:
        logx("Processing", repo_full_name)
        with open(keywords_to_look_for, 'r') as f:
            keywords = f.readlines()
        keywords = list([k.strip() for k in keywords])
        rd = RepoDownloader(repositories_path)
        cdg = KeywordFinder(keywords, result_location, bucket_proxy)

        repo_paths = rd.download_repos([repo_full_name])
        cdg.mine_repo(segments[0], segments[1], repo_paths[0])
        logx("DONE", repo_full_name)
        time.sleep(5)
        rd.remove_repos([repo_full_name])
    except Exception as e:
        logging.exception(e)
        logx('Failed to process repo', repo_full_name)



if __name__ == '__main__':
    start_time = time.time()
    logx('Starting...')
    logx(start_time)
    main()
    logx('===================')
    logx('=======DONE========')
    logx('===================')
    logx("--- %s seconds ---" % (time.time() - start_time))
    logx('===================')
    logx('===================')
    logx('===================')
    
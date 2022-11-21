import shutil
from git import Repo
from os import path

from src.utils.constants import github_home
from src.utils.utils import info_log


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
                info_log('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)
            elif force:
                self.remove_repos([repo_full_name])
                info_log('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)

        return paths


    def remove_repos(self, repo_list):
        for repo_full_name in repo_list:
            segments = repo_full_name.split('/')
            repo_path = path.join(self._path_to_store, segments[0], segments[1])
            if(path.exists(repo_path)):
                info_log('RepoDownloader', 'Removing repo', repo_path)
                try:    
                    # watch out for PermissionError: [WinError 5] Access is denied
                    shutil.rmtree(repo_path, ignore_errors=True)
                except Exception as e:
                    info_log(e)

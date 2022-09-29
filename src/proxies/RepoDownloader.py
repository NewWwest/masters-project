import shutil
from git import Repo
from os import path

from src.utils.constants import github_home


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
                print('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)
            elif force:
                self.remove_repos([repo_full_name])
                print('RepoDownloader', 'Downloading new repo', repo_path, repo_link)
                Repo.clone_from(repo_link, repo_path)

        return paths


    def remove_repos(self, repo_list):
        for repo_full_name in repo_list:
            segments = repo_full_name.split('/')
            repo_path = path.join(self._path_to_store, segments[0], segments[1])
            if(path.exists(repo_path)):
                print('RepoDownloader', 'Removing repo', repo_path)
                try:    
                    # watch out for PermissionError: [WinError 5] Access is denied
                    shutil.rmtree(repo_path, ignore_errors=True)
                except Exception as e:
                    print(e)


def main():
    npm_repos = [ 'moment/moment','lodash/lodash','eslint/eslint','webpack/webpack-dev-server','caolan/async']
    pypi_repos = ['psf/requests', 'numpy/numpy','django/django', 'python-pillow/Pillow','scipy/scipy']
    mvn_repos = ['junit-team/junit4','google/guava','spring-projects/spring-framework','h2database/h2database','google/gson']
    custom_repos = ['influxdata/telegraf', 'tensorflow/tensorflow']
    repos = npm_repos + pypi_repos + mvn_repos + custom_repos

    rd = RepoDownloader('/d/data/repos')
    rd.download_repos(repos)
    rd.remove_repos([repos])


if __name__ == '__main__':
    main()



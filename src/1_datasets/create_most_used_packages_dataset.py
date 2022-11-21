#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import requests
import json
import time
import pandas as pd

from src.proxies.GitHubProxy import GithubProxy
from src import secrets
from src.utils.constants import github_api

result_file = r'most_popular_{ecosystem}_packages.csv'
libraries_io_url = 'https://libraries.io/api/search?platforms={platform}&sort=rank&page={page}&api_key={apikey}'
aprox_number_of_repos = 1000
ecosystems = ['npm', 'maven', 'pypi']

def do_get(url):
    # Use do get instead of directly using requests.get to enforce the API limits
    time.sleep(1)
    return requests.get(url)


def main(ecosystem, limit):
    ghProxy = GithubProxy()
    repositories = set()
    result = []
    libraries_ecosystem_url = libraries_io_url.replace('{platform}', ecosystem).replace('{apikey}', secrets.libraries_io_token)
    page = 1
    while True:
        if len(repositories) >= limit:
            break

        print(page)
        response = do_get(libraries_ecosystem_url.replace('{page}', str(page)))
        page+=1
        if response.ok:
            data = json.loads(response.text)
            for x in data:
                if 'repository_url' in x and x['repository_url'] != None and len(x['repository_url']) > 0:
                    print(x['repository_url'])
                    if x['repository_url'] in repositories:
                        continue
                    if '/github.com/' not in x['repository_url']:
                        continue


                    segments = x['repository_url'].split('/')
                    github_repo_url = f'{github_api}/repos/{segments[3]}/{segments[4]}'
                    gh_respnse = ghProxy.do_get(github_repo_url)
                    if gh_respnse != None and gh_respnse.ok:
                        repo = json.loads(gh_respnse.text)
                        temp = {
                            "full_name": repo['full_name'],
                            'id':repo['id'],
                            'url':repo['url'],
                            'stars':repo['stargazers_count']
                            }
                        repositories.add(x['repository_url'])
                        result.append(temp)
                    else:
                        print('Request to GitHub Failed')
                        print(gh_respnse.text)

                    if len(repositories) >= limit:
                        break
        else:
            print('Request to libraires.io Failed')
            print(response.text)

    df = pd.DataFrame(result)
    df.to_csv(result_file.replace('{ecosystem}', ecosystem), index=False)


if __name__ == '__main__':
    for ecosystem in ecosystems:
        main(ecosystem, aprox_number_of_repos)


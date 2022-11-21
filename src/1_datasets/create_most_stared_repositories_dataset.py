#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import urllib.parse
import pandas as pd

from src.proxies.GitHubProxy import GithubProxy
from src.utils.utils import get_files_in_from_directory
from src.utils.constants import github_api


checkpoints_directory = r'results\most_stared_checkpoints'
result_file = r'most_starred_repositories.csv'
aprox_number_of_repos = 10000
step = 50

ghProxy = GithubProxy()

def fetch_repos(limit, stars_down, stars_up):
    q = urllib.parse.quote_plus(f'stars:{stars_down}..{stars_up} fork:true')	 
    search_repos_endpoint = f'{github_api}/search/repositories?q={q}&sort=stars&order=desc'

    repositories = ghProxy.iterate_search_endpoint(search_repos_endpoint, limit)
    
    result = []
    for repo in repositories:
        result.append({'full_name': repo['full_name'], 'id':repo['id'], 'url': repo['url'], 'stars': repo['stargazers_count']})

    df = pd.DataFrame(result)
    df.to_csv(f'{checkpoints_directory}/results-{stars_down}-{stars_up}.csv', index=False)
    return df


def get_repos(limit, output_file):
    max_stars = 10_000_000
    min_stars = 20_000
    count = 0
    while True:
        print(f'Fetching repos between {min_stars} and {max_stars}')
        repos = fetch_repos(1000, min_stars, max_stars)
        count += repos.shape[0]
        max_stars = min_stars
        min_stars = min_stars-step
        if count >= limit:
            break

    consolidate_files(output_file)


def consolidate_files(output_file):
    data_files = get_files_in_from_directory(checkpoints_directory)
    dataframes = [pd.read_csv(x) for x in data_files]

    df = pd.concat(dataframes)
    df.drop_duplicates(subset=['full_name'], keep=False, inplace=True)
    df.sort_values('stars', ascending=False, inplace=True)

    df.to_csv(output_file, index=False)
    return df


def main():
    get_repos(aprox_number_of_repos, result_file)

if __name__ == '__main__':
    main()

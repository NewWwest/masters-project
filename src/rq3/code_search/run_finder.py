import sys
sys.path.insert(0, r'D:\Projects\aaa')

from multiprocessing import Pool
import pandas as pd
import time
import os
import json


from src.proxies.RepoDownloader import RepoDownloader
from src.rq3.code_search.setBasedKeywordFinder import KeywordFinder

cpus = 25
repositories_path = '/repolist3'

result_location = r'results\keywords_find'
repositories_to_process = r'src\rq3\repos_to_process.csv'
keywords_to_look_for = r'src\rq3\keywords.csv'

def main():
    repos_df = pd.read_csv(repositories_to_process)
    repos = list(repos_df['full_name'])
    repos.sort()

    with Pool(cpus) as p:
        p.map(mine_a_repo, repos, chunksize=1)


def mine_a_repo(repo_full_name):
    segments = repo_full_name.split('/')

    try:
        print("Processing", repo_full_name)
        keywords_df = pd.read_csv(keywords_to_look_for)
        keywords = list(keywords_df['word'])
        rd = RepoDownloader(repositories_path)
        cdg = KeywordFinder(keywords, result_location)

        repo_paths = rd.download_repos([repo_full_name])
        cdg.mine_repo(segments[0], segments[1], repo_paths[0])
        with open(f'{result_location}/processing-result-{segments[0]}-{segments[1]}', 'w') as f:
            json.dump({'success': True}, f)

        print("DONE", repo_full_name)
        time.sleep(5)
        rd.remove_repos([repo_full_name])
    except Exception as e:
        with open(f'{result_location}/processing-result-{segments[0]}-{segments[1]}', 'w') as f:
            json.dump({'success': False}, f)
        print('Failed to process repo', repo_full_name)
        print(e)


# def remove_results_for_repo(owner, repo):
#     files = os.listdir(result_location) 

#     for_this_repo = [f for f in files if f.startswith(f'keywords_in_diff-{owner}-{repo}')]
#     for x in for_this_repo:
#         os.remove(x) 

#     result_file = f'{result_location}/processing-result-{owner}-{repo}'
#     if os.path.exists(result_file):
#         os.remove(x) 

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    
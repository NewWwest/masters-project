import sys
sys.path.insert(0, r'D:\Projects\aaa')

import time
import random
from multiprocessing import Pool
from src.loaders.SecurityRelatedCommitsLoader import SecurityRelatedCommitsLoader
from src.proxies.RepoDownloader import RepoDownloader

# Configure the miner you want to use
from src.mining.CodeMiners.RollingWindowMiner import RollingWindowMiner as TheMiner
cpus = 20
minimal_number_of_commits_in_repo = 5 
repositories_path = '/repolist2'
path_to_security_commits = r'src\rq1\final_results\security_relevant_commits.csv'


def main():
    cs = SecurityRelatedCommitsLoader(path_to_security_commits)
    sorted_repos = cs.get_repos_with_atleastn_fix_commits(minimal_number_of_commits_in_repo)
    sorted_repos = sorted_repos[1:]

    with Pool(cpus) as p:
        p.map(mine_a_repo, sorted_repos, chunksize=1)


def mine_a_repo(repo_full_name):
    try:
        time.sleep(random.randint(1, 7))
        print("MINING", repo_full_name)
        rd = RepoDownloader(repositories_path)
        cs = SecurityRelatedCommitsLoader(path_to_security_commits)
        cdg = TheMiner()
        segments = repo_full_name.split('/')

        repo_paths = rd.download_repos([repo_full_name])
        cdg.mine_repo(segments[0], segments[1], repo_paths[0], cs.shas_for_repo(repo_full_name))
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
    
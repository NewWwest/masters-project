import os
from multiprocessing import Pool

import sys
sys.path.insert(0, r'D:\Projects\2022')
from src.linked_commit_investigation.CommitSelector import CommitSelector
from src.linked_commit_investigation.CommitMiner import CommitMiner
from src.linked_commit_investigation.RepoDownloader import RepoDownloader
from src.linked_commit_investigation.FeatureCalculator import FeatureCalculator
from src.linked_commit_investigation.CommitPreselector import CommitPreselector 

cpus = 20
limit = 5
features_cache_location = r'D:\Projects\2022\results\checkpoints1'
repositories_path = '/d/Projects/data/repos3'
mapping_file = 'mapping_vulnerabilities_to_commits.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec-star.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec.json'

def mine_most_common_projects():
    cs = CommitSelector(mapping_file)
    sorted_repos = cs.get_repos_with_atleastn_fix_commits(limit)

    with Pool(cpus) as p:
        p.map(mine_a_repo, sorted_repos, chunksize=1)


def mine_a_repo(repo_full_name):
    print("Processing", repo_full_name)
    if repo_full_name == 'torvalds/linux':
        print('Cloning linux takes too long :x')
        return
        
    try:
        # segments = repo_full_name.split('/')
        # pathx = f'{features_cache_location}/commit-level-{segments[0]}-{segments[1]}.csv'
        # if os.path.exists(pathx):
        #     print("Skipping", repo_full_name)
        #     return

        print("MINING", repo_full_name)
        cm = CommitMiner(repositories_path)
        rd = RepoDownloader(repositories_path)
        cs = CommitSelector(mapping_file)
        cps = CommitPreselector()
        fc = FeatureCalculator()

        # rd.download_repos([repo_full_name])
        # cm.mine_repo_2(repo_full_name, fix_commits=cs.fix_commits_by_repo[repo_full_name], partial_mine=True)
        cps.filter_commits_for_repo(repo_full_name)
        fc.process_repo(repo_full_name)
        cs.select_commits_for_repo(repo_full_name)
        # rd.remove_repos([repo_full_name])

    except Exception as e:
        print('Failed to process repo', repo_full_name)
        print(e)


if __name__ == '__main__':
    mine_most_common_projects()
    pass
    
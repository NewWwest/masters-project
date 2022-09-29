import os
import pandas as pd
import json
import time 
import sys 

sys.path.insert(0, r'D:\Projects\2022')
print( sys.path)
from src.linked_commit_investigation.CommitSelector import CommitSelector
from src.linked_commit_investigation.CommitMiner import CommitMiner
from src.linked_commit_investigation.RepoDownloader import RepoDownloader

repositories_path = '/d/Projects/data/repos'
mapping_file = 'mapping_vulnerabilities_to_commits.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec-star.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec.json'

def mine_most_common_projects(skip_linux=True):
    cm = CommitMiner(repositories_path)
    rd = RepoDownloader(repositories_path)
    cs = CommitSelector(mapping_file)

    with_fix_count = []
    for x in cs.fix_commits_by_repo:
        with_fix_count.append((x, len(cs.fix_commits_by_repo[x])))

    with_fix_count = sorted(with_fix_count, key=lambda x: x[1], reverse=True)

    start = 2
    for x in range(start, len(with_fix_count)):
        try:
            repo_full_name = with_fix_count[x][0]
            print("MINING", repo_full_name)
            time.sleep(2)
            rd.download_repos([repo_full_name])
            cm.mine_repo_2(repo_full_name, fix_commits=cs.fix_commits_by_repo[repo_full_name], partial_mine=True)

        except Exception as e:
            print('Failed to process repo', repo_full_name)
            print(e)


if __name__ == '__main__':
    mine_most_common_projects()
    pass
    
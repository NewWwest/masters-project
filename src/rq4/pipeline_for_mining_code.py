from multiprocessing import Pool

import sys
sys.path.insert(0, r'D:\Projects\2022')
from src.linked_commit_investigation.CommitSelector import CommitSelector
from src.linked_commit_investigation.RepoDownloader import RepoDownloader
from src.linked_commit_investigation.CodeDataGenerator import CodeDataGenerator 

cpus = 10
limit = 5
repos_to_process = 20
data_location = r'D:\Projects\2022\results\codepipeline_ch5'
repositories_path = '/d/Projects/data/repos3'
# mapping_file = 'mapping_vulnerabilities_to_commits.json'
mapping_file = 'D:\\Projects\\2022\\sec-star.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec.json'

def mine_most_common_projects():
    cs = CommitSelector(mapping_file)
    sorted_repos = cs.get_repos_with_atleastn_fix_commits(limit)
    sorted_repos = sorted_repos[:repos_to_process]

    with Pool(cpus) as p:
        p.map(mine_a_repo, sorted_repos, chunksize=1)


def mine_a_repo(repo_full_name):
    print("Processing", repo_full_name)
    if repo_full_name == 'torvalds/linux':
        print('Cloning linux takes too long :x')
        return
        
    try:
        print("MINING", repo_full_name)
        rd = RepoDownloader(repositories_path)
        cdg = CodeDataGenerator(data_location, ['.java'])

        repo_paths = rd.download_repos([repo_full_name])
        cdg.mine_repo(repo_full_name, repo_paths[0])

        rd.remove_repos([repo_full_name])
        print("DONE", repo_full_name)

    except Exception as e:
        print('Failed to process repo', repo_full_name)
        print(e)


if __name__ == '__main__':
    mine_most_common_projects()
    pass
    
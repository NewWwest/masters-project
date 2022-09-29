from pydriller import RepositoryMining, GitRepository
from datetime import datetime
import regex as re
import json

start_year = 2017 
checkpoint_frequency = 200


class KeywordFinder:
    def __init__(self, keywords, checkpoints_directory):
        self.checkpoints_directory = checkpoints_directory
        self.keywords = [re.compile(r'(?i)\b' + k + r'\b') for k in keywords]

    def mine_repo(self, owner: str, repo: str, repo_path: str):
        processed_commits = 0
        result = []
        for commit in RepositoryMining(repo_path, since=datetime(start_year, 1, 1)).traverse_commits():
            processed_commits += 1
            for changeFile in commit.modifications:
                tokens = self.split_regex.split(changeFile.diff.lower())
                for token in tokens:
                    if token in self.keywords:
                        res = {
                            'owner':owner,
                            'repo':repo,
                            'hash':commit.hash,
                            'path':changeFile.new_path if changeFile.new_path != None else changeFile.old_path,
                            'keyword': token
                        }
                        result.append(res)


        self.save_to_checkpoint(result, owner, repo, 'final', processed_commits)
        processed_commits = 0
        result = []


    def save_to_checkpoint(self, data, owner, repo, hash, processed_commits):
        temp_data = {
            'processed_commits':processed_commits,
            'data': data
        }
        with open(f'{self.checkpoints_directory}/keywords_in_diff-{owner}-{repo}-{hash}.json', 'w')  as f:
            json.dump(temp_data, f)


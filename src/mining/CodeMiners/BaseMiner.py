from abc import abstractmethod
from pydriller import RepositoryMining, Commit
import pandas as pd
import json

class BaseMiner:
    def __init__(self, sample_encodder):
        self.sample_encodder = sample_encodder


    @abstractmethod
    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        pass

    def save(self, checkpoints_directory, owner, repo, commit, label_security_related, commit_data):
        if commit_data == None or len(commit_data) == 0:
            return

        flag = 'positive' if label_security_related else 'background'
        sha = commit.hash
        path_to_new_file = f'{checkpoints_directory}/{flag}-samples-{owner}-{repo}-{sha}.json'
        with open(path_to_new_file, 'w') as f:
            json.dump(commit_data, f)

        return [path_to_new_file]
            

    def save_and_tokenize(self, checkpoints_directory, owner, repo, commit, label_security_related, commit_data):
        if commit_data == None or len(commit_data) == 0:
            return

        datafiles1 = self.save(checkpoints_directory, owner, repo, commit, label_security_related, commit_data)
            
        flag = 'positive' if label_security_related else 'background'
        sha = commit.hash
        commit_data_encodded = []
        for sample in commit_data:
            tokens = self.sample_encodder.process_sample(sample)
            res1 = {
                    'commit_id': sample['commit_id'],
                    'sample_type': sample['sample_type'],
                    'file_name': sample['file_name'],
                    'is_security_related': sample['is_security_related'],
                    'commit_sample': tokens
                }
            commit_data_encodded.append(res1)

        path_to_new_file =f'{checkpoints_directory}/{flag}-encodings-{owner}-{repo}-{sha}.json'
        with open(path_to_new_file, 'w') as f:
            json.dump(commit_data_encodded, f)

        return [path_to_new_file] + datafiles1


from abc import abstractmethod
from pydriller import RepositoryMining, Commit
import pandas as pd
import json

class BaseMiner:
    def __init__(self, checkpoints_directory, sample_encodder):
        self.checkpoints_directory = checkpoints_directory
        self.sample_encodder = sample_encodder


    @abstractmethod
    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        pass

    def mine_commit2(self, owner, repo, commit: Commit, label_security_related):
        try:
            commit_data = self._mine_commit(owner, repo, commit, label_security_related)
            if commit_data != None and len(commit_data) > 0:
                flag = 'positive' if label_security_related else 'background'
                sha = commit.hash
                with open(f'{self.checkpoints_directory}/{flag}-samples-{owner}-{repo}-{sha}.json', 'w') as f:
                    json.dump(commit_data, f)
                
                commit_data_encodded = []
                for sample in commit_data:
                    tokens = self.sample_encodder.process_sample(sample)
                    res1 = {
                        'commit_id': sample['commit_id'],
                        'file_name': sample['file_name'],
                        'is_security_related': sample['is_security_related'],
                        'commit_sample': tokens
                    }
                    commit_data_encodded.append(res1)

                with open(f'{self.checkpoints_directory}/{flag}-encodings-{owner}-{repo}-{sha}.json', 'w') as f:
                    json.dump(commit_data_encodded, f)

        except Exception as e:
            print('Processing commit failed', owner, repo, commit.hash)
            print(e)

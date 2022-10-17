from pydriller import GitRepository, Commit
from typing import Iterable
from pydriller import GitRepository

from src.mining.CodeMiners.BaseMiner import BaseMiner

class AddedCodeMiner(BaseMiner):
    def __init__(self, results_dir, sample_encodder, valid_extensions):
        super().__init__(results_dir, sample_encodder)
        self.valid_extensions = valid_extensions


    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        commit_id = f'{owner}/{repo}/{commit.hash}'
        commit_first_line = commit.msg.split('\n')[0]
        commit_title = commit_first_line[0:min(72,len(commit_first_line))]

        changeFiles = []
        for changeFile in commit.modifications:
            safe_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path

            if safe_path.split('.')[-1] not in self.valid_extensions:
                continue

            if 'added' not in changeFile.diff_parsed:
                continue

            lines = [x[1] for x in changeFile.diff_parsed['added']]
            res1 = {
                'commit_id': commit_id,
                'file_name': safe_path,
                'is_security_related': label_security_related,
                'commit_title': commit_title,
                'commit_sample': '\n'.join(lines)
            }
            changeFiles.append(res1)
        
        return changeFiles


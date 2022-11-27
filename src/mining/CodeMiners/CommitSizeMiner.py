from pydriller import Commit

from src.mining.CodeMiners.BaseMiner import BaseMiner

class CommitSizeMiner(BaseMiner):
    def __init__(self, results_dir, extensions_to_ignore):
        super().__init__(None)
        self.checkpoints_directory = results_dir
        self.extensions_to_ignore = set(extensions_to_ignore) if extensions_to_ignore else None


    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        commit_id = f'{owner}/{repo}/{commit.hash}'

        changeFiles = []
        for changeFile in commit.modifications:
            safe_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
            
            if self.extensions_to_ignore != None and changeFile.filename.split('.')[-1] in self.extensions_to_ignore:
                continue

            res1 = {
                'id': commit_id,
                'sample_type': 'CommitSizeMiner',
                'file_name': safe_path,
                'commit_size': changeFile.removed + changeFile.added,
                'is_security_related': label_security_related,
            }
            changeFiles.append(res1)
        
        return self.save(self.checkpoints_directory, owner, repo, commit, label_security_related, changeFiles)

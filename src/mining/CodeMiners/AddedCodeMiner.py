from typing import Iterable
from pydriller import GitRepository, Commit
import random
from typing import Iterable
from pydriller import GitRepository
import random

from src.mining.CodeMiners.BaseMiner import BaseMiner

start_year = 2010
checkpoint_frequency = 250
RATIO = 50
diff_checkpoints_directory = r'results\added_code_miner_results'

valid_extensions = set()
valid_extensions.update(['java', 'scala', 'kt', 'swift'])
valid_extensions.update(['js', 'jsx', 'ts'])
valid_extensions.update(['py', 'ipynb'])
valid_extensions.update(['cpp', 'c', 'cs', 'cshtml', 'sql', 'r', 'vb', 'php'])

class AddedCodeMiner(BaseMiner):
    def __init__(self):
        random.seed(42)
        super().__init__(diff_checkpoints_directory)

    def mine_repo(self, owner: str, repo: str, repo_path: str, fix_commits: Iterable):
        self.fix_commits_set = set(fix_commits)
        gr = GitRepository(repo_path)
        all_commit_count = gr.total_commits()
        self.probability_of_mine = len(self.fix_commits_set)*RATIO / all_commit_count
        self._iterate_commits(owner, repo, repo_path)


    def _should_mine_commit(self, commit):
        return commit.hash in self.fix_commits_set or random.random() <= self.probability_of_mine


    def _mine_commit(self, owner, repo, commit: Commit):
        commit_id = f'{owner}/{repo}/{commit.hash}'
        commit_first_line = commit.msg.split('\n')[0]
        commit_title = commit_first_line[0:min(72,len(commit_first_line))]

        changeFiles = []
        for changeFile in commit.modifications:
            save_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
            ext = save_path.split('.')[-1]
            if ext not in valid_extensions:
                continue
            if 'added' not in changeFile.diff_parsed:
                continue

            lines = [x[1] for x in changeFile.diff_parsed['added']]
            res1 = {
                'commit_id': commit_id,
                'file_name': save_path,
                'is_security_related': commit.hash in self.fix_commits_set,
                'commit_title': commit_title,
                'commit_sample': '\n'.join(lines)
            }
            changeFiles.append(res1)
        
        return changeFiles


import random

from src.mining.CodeMiners.BaseMiner import BaseMiner

class RollingWindowMiner(BaseMiner):
    def __init__(self, results_location, sample_encodder, rolling_line_window, max_samples_per_commit, valid_extensions, extensions_to_ignore):
        super().__init__(sample_encodder)
        self.checkpoints_directory = results_location
        self.rolling_line_window = rolling_line_window
        self.max_samples_per_commit = max_samples_per_commit
        self.valid_extensions = set(valid_extensions) if valid_extensions else None
        self.extensions_to_ignore = set(extensions_to_ignore) if extensions_to_ignore else None


    def _mine_commit(self, owner, repo, commit, label_security_related):
        lines_and_files_all = []
        files_all = []
        filenames_all = []

        i = 0
        for changeFile in commit.modifications:
            if changeFile.new_path == None or changeFile.source_code == None:
                continue

            if self.valid_extensions != None and changeFile.filename.split('.')[-1] not in self.valid_extensions:
                continue

            if self.extensions_to_ignore != None and changeFile.filename.split('.')[-1] in self.extensions_to_ignore:
                continue

            filenames_all.append(changeFile.new_path)
            files_all.append(changeFile.source_code.split('\n'))
            lines_and_files = [(added_in_diff[0], i) for added_in_diff in changeFile.diff_parsed['added']]
            lines_and_files_all += lines_and_files
            i+=1

        if len(lines_and_files_all) == 0:
            return None

        commit_id = f'{owner}/{repo}/{commit.hash}'
        commit_first_line = commit.msg.split('\n')[0]
        commit_title = commit_first_line[0:min(72,len(commit_first_line))]
        picked_lines = random.sample(lines_and_files_all, min(self.max_samples_per_commit, len(lines_and_files_all)))

        commit_files = []
        for picked_line in picked_lines:
            line_number = picked_line[0]
            code = files_all[picked_line[1]]
            sampled_code = code[max(0, line_number-self.rolling_line_window):min(line_number+self.rolling_line_window, len(code))]
            res1 = {
                'commit_id': commit_id,
                'sample_type': 'RollingWindowMiner',
                'file_name': filenames_all[picked_line[1]],
                'is_security_related': label_security_related,
                'commit_title': commit_title,
                'commit_sample': '\n'.join(sampled_code)
            }
            commit_files.append(res1)

        return self.save_and_tokenize(self.checkpoints_directory, owner, repo, commit, label_security_related, commit_files)



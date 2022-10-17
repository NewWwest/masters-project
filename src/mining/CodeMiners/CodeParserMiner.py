import random

from src.mining.CodeMiners.BaseMiner import BaseMiner
from src.mining.CodeMiners.GumTreeProxy import GumTreeProxy



max_samples_per_commit = 100


class CodeParserMiner(BaseMiner):
    def __init__(self, results_location, sample_encodder, valid_extensions):
        super().__init__(results_location, sample_encodder)
        # self.valid_extensions = valid_extensions
        self.valid_extensions = set(['java'])

    def _mine_commit(self, owner, repo, commit, label_security_related):
        lines_and_files_all = []
        files_all = []
        filenames_all = []

        i = 0
        for changeFile in commit.modifications:
            if changeFile.new_path == None or changeFile.source_code == None or changeFile.old_path == None or changeFile.source_code_before == None:
                continue

            if changeFile.new_path.split('.')[-1] not in self.valid_extensions:
                continue

            results = GumTreeProxy.get_parsed_code_data(changeFile.source_code_before, changeFile.source_code, changeFile.filename)
            if results == None:
                continue
            ast_data = results[0]
            change_data = results[1]

            # TODO use the data

            i+=1

        if len(lines_and_files_all) == 0:
            return None


        return None



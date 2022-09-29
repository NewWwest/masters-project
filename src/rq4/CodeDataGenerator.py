from pydriller import RepositoryMining 
from transformers import RobertaTokenizer
from datetime import datetime
import shutil
import unicodedata
import regex as re
import pandas as pd
import os
import json


start_year = 2015
regex = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

class CodeDataGenerator:
    def __init__(self, output_directory, extensions_to_mine):
        self._output_directory = output_directory
        self._extensions_to_mine = set(extensions_to_mine)

    def mine_repo(self, repo_full_name: str, repo_path: str):
        segments = repo_full_name.split('/')
        self.ensure_directory(f'{self._output_directory}\\{segments[0]}')
        self.ensure_directory(f'{self._output_directory}\\{segments[0]}\\{segments[1]}')

        for commit in RepositoryMining(repo_path, since=datetime(start_year,1,1)).traverse_commits():
            path = f'{self._output_directory}\\{segments[0]}\\{segments[1]}\\{commit.hash}'
            self.ensure_directory(path)

            commit_files = []
            file_index = -1
            for changeFile in commit.modifications:
                _, file_extension = os.path.splitext(changeFile.filename)
                if file_extension not in self._extensions_to_mine:
                    continue

                file_index += 1
                file_title = f'{file_index}-{changeFile.filename}'
                commit_files.append(file_title)

                new_code = ''
                new_code_tokenized = []
                
                old_code = ''
                old_code_tokenized = []

                new_lines = [x[0] for x in changeFile.diff_parsed['added']] 
                deleted_lines = [x[0] for x in changeFile.diff_parsed['deleted']] 

                with open(f'{path}\\new_{file_title}.lines.json', 'w') as f:
                    json.dump(new_lines, f)
                with open(f'{path}\\old_{file_title}.lines.json', 'w') as f:
                    json.dump(deleted_lines, f)

                new_code = changeFile.source_code if changeFile.source_code else ''
                old_code = changeFile.source_code_before if changeFile.source_code_before else ''

                with open(f'{path}\\new_{file_title}', 'w', encoding='UTF-8') as f:
                    f.write(new_code)
                with open(f'{path}\\old_{file_title}', 'w', encoding='UTF-8') as f:
                    f.write(old_code)
                

                new_code_tokenized = [self.tokenize_line(line) for line in new_code.splitlines()]
                old_code_tokenized = [self.tokenize_line(line) for line in old_code.splitlines()]

                with open(f'{path}\\new_{file_title}.tokens.json', 'w') as f:
                    json.dump(new_code_tokenized, f)
                with open(f'{path}\\old_{file_title}.tokens.json', 'w') as f:
                    json.dump(old_code_tokenized, f)
            
            if len(commit_files) > 0:
                with open(f'{path}\\commit.json', 'w') as f:
                    commit_data = {
                        'commit_title': commit.msg.split('\n')[0],
                        'commits': commit_files
                    }
                    json.dump(commit_data, f)
            else:
                try:    
                    # watch out for PermissionError: [WinError 5] Access is denied
                    shutil.rmtree(path)
                except Exception as e:
                    print(e)


    def ensure_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


    def tokenize_line(self, line):
        added_line2 = str(unicodedata.normalize('NFKD', line).encode('ascii', 'ignore'))
        added_line3 = regex.split(added_line2) + ['\n']
        added_line4 = ' '.join(added_line3)
        return tokenizer.tokenize(added_line4)


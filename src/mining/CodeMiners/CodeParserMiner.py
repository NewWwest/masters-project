# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
import random
from src.mining.CodeMiners.BaseMiner import BaseMiner
from src.mining.CodeMiners.GumTreeProxy import GumTreeProxy
from pydriller import RepositoryMining, Commit, Modification


handlable_extensions = ['c', 'cpp', 'cs', 'java', 'js', 'py', 'py3']


class CodeParserMiner(BaseMiner):
    def __init__(self, ast_results_location, actions_results_location, sample_encodder, max_samples_per_commit, valid_extensions, extensions_to_ignore):
        super().__init__(sample_encodder)
        self.valid_extensions = valid_extensions.intersection(set(handlable_extensions))
        self.extensions_to_ignore = set(extensions_to_ignore) if extensions_to_ignore else None
        self.max_samples_per_commit = max_samples_per_commit

        self.ast_results_location = ast_results_location
        self.actions_results_location = actions_results_location

    def _mine_commit(self, owner, repo, commit: Commit, label_security_related):
        # at least one change per file.
        # then up to 100 samples accroding to the largest files
        valid_files_to_changed_lines = {}
        for changeFile in commit.modifications:
            if changeFile.new_path == None or changeFile.source_code == None or changeFile.old_path == None or changeFile.source_code_before == None:
                continue

            if len(changeFile.source_code) > 1_000_000 or len(changeFile.source_code_before) > 1_000_000:
                continue

            if self.valid_extensions != None and changeFile.filename.split('.')[-1] not in self.valid_extensions:
                continue

            if self.extensions_to_ignore != None and changeFile.filename.split('.')[-1] in self.extensions_to_ignore:
                continue

            valid_files_to_changed_lines[changeFile.new_path] = changeFile.added

        added_lines_sum = sum([v for k,v in valid_files_to_changed_lines.items()])
        samples_to_distribute = self.max_samples_per_commit - len(valid_files_to_changed_lines)

        ast_samples = []
        action_samples = []

        for changeFile in commit.modifications:
            if changeFile.new_path not in valid_files_to_changed_lines:
                continue

            results = GumTreeProxy.get_parsed_code_data(changeFile.source_code_before, changeFile.source_code, changeFile.filename)
            if results == None:
                continue

            ast_data = results[0]
            change_data = results[1]

            num_of_samples_to_extract = 1
            num_of_samples_to_extract += int(samples_to_distribute * changeFile.added/added_lines_sum)
            file_ast_samples = self._get_ast_samples(changeFile, ast_data, num_of_samples_to_extract)
            file_action_samples = self._get_action_samples(changeFile, change_data, num_of_samples_to_extract)
            safe_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
            commit_id = f'{owner}/{repo}/{commit.hash}'
            commit_first_line = commit.msg.split('\n')[0]
            commit_title = commit_first_line[0:min(72,len(commit_first_line))]

            for x in file_ast_samples:
                res1 = {
                    'commit_id': commit_id,
                    'sample_type': 'CodeParserMiner_ast',
                    'file_name': safe_path,
                    'is_security_related': label_security_related,
                    'commit_title': commit_title,
                    'commit_sample': x
                }
                ast_samples.append(res1)

            for x in file_action_samples:
                res1 = {
                    'commit_id': commit_id,
                    'sample_type': 'CodeParserMiner_edit',
                    'file_name': safe_path,
                    'is_security_related': label_security_related,
                    'commit_title': commit_title,
                    'commit_sample': x
                }
                action_samples.append(res1)

        files1 = self.save_and_tokenize(self.ast_results_location, owner, repo, commit, label_security_related, ast_samples)
        if files1 == None:
            files1 = []
        files2 = self.save_and_tokenize(self.actions_results_location, owner, repo, commit, label_security_related, action_samples)
        if files2 == None:
            files2 = []

        return files1 + files2


    def _get_ast_samples(self, changeFile: Modification, ast_data, num_of_samples_to_extract):
        if len(changeFile.diff_parsed['added']) == 0:
            return[]

        samples = []
        lines = changeFile.source_code.split('\n')
        start_bytes_of_each_line = self._get_start_bytes_of_each_line(lines)
        self._annotate_ast_with_start_lines(start_bytes_of_each_line, ast_data)
        self._reject_nodes_under_one_line_statements(ast_data)
        for i in range(num_of_samples_to_extract):
            start = random.choice(changeFile.diff_parsed['added'])
            end = random.choice(changeFile.diff_parsed['added'])
            if start[0] > end[0]:
                temp = end
                end = start
                start = temp

            path_to_start = self._path_to_line(ast_data, start[0])
            path_to_end = self._path_to_line(ast_data, end[0])
            path = self._find_path(path_to_start, path_to_end)
            start_lines = list(set([x['StartLine'] for x in path]))
            start_lines.sort()
            selected_lines = []
            for x in start_lines:
                selected_lines.append(lines[x])

            samples.append('\n'.join(selected_lines))
        return samples


    def _get_start_bytes_of_each_line(self, lines):
        current = 0
        new_line_size = len('\n'.encode('utf-8'))
        start_bytes_of_each_line = []
        for l in lines:
            start_bytes_of_each_line.append(current)
            line_length = len(l.encode('utf-8')) + new_line_size
            current += line_length
            
        return start_bytes_of_each_line


    def _annotate_ast_with_start_lines(self, start_bytes_of_each_line, ast_data):
        ast_data['StartLine'] = 0
        for i in range(len(start_bytes_of_each_line)):
            if ast_data['StartByte'] < start_bytes_of_each_line[i]:
                ast_data['StartLine'] = i - 1
                break

        ast_data['EndLine'] = ast_data['StartLine']
        for i in range(ast_data['StartLine'], len(start_bytes_of_each_line)):
            if ast_data['EndByte'] >= start_bytes_of_each_line[i]:
                ast_data['EndLine'] = i
            else:
                break

        for ch in ast_data['Children']:
            self._annotate_ast_with_start_lines(start_bytes_of_each_line, ch)


    def _reject_nodes_under_one_line_statements(self, ast):
        if ast['StartLine'] == ast['EndLine']:
            ast['Children'] = []
        else:
            for ch in ast['Children']:
                self._reject_nodes_under_one_line_statements(ch)


    def _path_to_line(self, ast, line_number):
        stack = [ast]
        path_down = []
        while len(stack) > 0:
            current_tree = stack.pop()
            if current_tree['StartLine'] <= line_number and line_number < current_tree['EndLine']: 
                path_down.append(current_tree)
                for ch in current_tree['Children']:
                    stack.append(ch)

        return path_down


    def _find_path(self, path_to_start, path_to_end):
        shortest_path = []
        for i in range(len(path_to_start)-1, 0, -1):
            found_shared = False
            for j in range(len(path_to_end)-1, 0, -1):
                if path_to_start[i] == path_to_end[j]:
                    found_shared = True
                    break
            if found_shared:
                shortest_path += path_to_end[j:]
            else:
                shortest_path.append(path_to_start[i])
                    
        return shortest_path


    def _get_action_samples(self, changeFile: Modification, change_data, num_of_samples_to_extract):
        if change_data['Children'] == None or len(change_data['Children']) == 0:
            return []  

        lines = changeFile.source_code.split('\n')
        start_bytes_of_each_line = self._get_start_bytes_of_each_line(lines)
        lines_of_added_statements = set()

        for action in change_data['Children']:
            if action['Label'] == 'insert-node':
                self._annotate_ast_with_start_lines(start_bytes_of_each_line, action)
                for x in range(action['StartLine'], action['EndLine']):
                    lines_of_added_statements.add(x)
            elif action['Label'] == 'delete-tree':
                pass

        if len(lines_of_added_statements) == 0:
            return []
        lines_of_added_statements = list(lines_of_added_statements)
        lines_of_added_statements.sort()

        sample_line_numbers = []
        current = [lines_of_added_statements[0]]
        for i in range(1, len(lines_of_added_statements)):
            current.append(lines_of_added_statements[i])
            if abs(lines_of_added_statements[i]-lines_of_added_statements[i-1]) != 1:
                sample_line_numbers.append(current)
                current = [lines_of_added_statements[i]]
            else:
                current.append(lines_of_added_statements[i])

        sample_line_numbers.append(current)

        samples = []
        for s in sample_line_numbers:
            samples.append([lines[x] for x in s])
        result = ['\n'.join(x) for x in samples]
        return result

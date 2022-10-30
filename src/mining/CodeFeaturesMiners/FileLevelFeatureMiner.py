from pydriller import Modification,ModificationType
from pydriller import Commit
from os import path
import src.mining.CodeFeaturesMiners.constants_and_configs as constants_and_configs

from src.mining.CodeFeaturesMiners.AbstractMiner import AbstractMiner


class FileLevelFeatureMiner(AbstractMiner):
    def __init__(self, feature_classes) -> None:
        super().__init__(feature_classes)


    def mine_commit(self, repo_full_name, changeFile: Modification, history, index):
        self.features = {}
        self._mine_labels(changeFile, history[index], repo_full_name)
        if 'basic' in self.feature_classes:
            self._mine_basic(changeFile)
        if 'code_analysis' in self.feature_classes:
            self._mine_code_analysis(changeFile)
        if 'bag_of_words' in self.feature_classes:
            self._mine_bag_of_words(changeFile)
        if 'history' in self.feature_classes and history != None and index != None:
            self._mine_history_based(changeFile, history, index)
        return self.features


    def _mine_labels(self, changeFile: Modification, commit:Commit, repo_full_name):
        path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
        self._add_feature('label_sha', commit.hash)
        self._add_feature('label_commit_date', commit.committer_date)
        self._add_feature('label_repo_full_name', repo_full_name)
        self._add_feature('label_filepath', path.lower())


    def _mine_basic(self, changeFile: Modification):
        filename = changeFile.filename.lower()
        _, file_extension = path.splitext(filename)
        file_extension = file_extension.lower()
        
        has_ecosystem_files = {k:False for k in constants_and_configs.code_extensions}
        for ecosystem in constants_and_configs.code_extensions:
            for extension in constants_and_configs.code_extensions[ecosystem]:
                if file_extension == extension:
                    has_ecosystem_files[ecosystem] = True
                    break
                
        for k,v in has_ecosystem_files.items():
            self._add_feature(f'has_{k}_code', v)

        is_add = changeFile.change_type == ModificationType.ADD
        self._add_feature('is_add', is_add)
        is_rename = changeFile.change_type == ModificationType.RENAME
        self._add_feature('is_rename', is_rename)
        is_delete = changeFile.change_type == ModificationType.DELETE
        self._add_feature('is_delete', is_delete)
        is_modify = changeFile.change_type == ModificationType.MODIFY
        self._add_feature('is_modify', is_modify)

        removed_lines_count = changeFile.removed if changeFile.removed != None else 0
        self._add_feature('removed_lines_count', removed_lines_count)
        added_lines_count = changeFile.added if changeFile.added != None else 0
        self._add_feature('added_lines_count', added_lines_count)
        changed_lines_count = removed_lines_count + added_lines_count
        self._add_feature('changed_lines_count', changed_lines_count)
        removed_lines_ratio = removed_lines_count/changed_lines_count if changed_lines_count>0 else float('nan')
        self._add_feature('removed_lines_ratio', removed_lines_ratio)
        added_lines_ratio = added_lines_count/changed_lines_count if changed_lines_count>0 else float('nan')
        self._add_feature('added_lines_ratio', added_lines_ratio)

        if 'added' in changeFile.diff_parsed and 'deleted' in changeFile.diff_parsed:
            added_lines_set = set([x[0] for x in changeFile.diff_parsed['added']])
            removed_lines_set = set([x[0] for x in changeFile.diff_parsed['deleted']])
            modified_lines_count = len(added_lines_set.intersection(removed_lines_set))
            self._add_feature('modified_lines_count', modified_lines_count)
            modified_lines_ratio = modified_lines_count / changed_lines_count if changed_lines_count>0 else float('nan')
            self._add_feature('modified_lines_ratio', modified_lines_ratio)

        file_size = float("nan")
        if changeFile.source_code != None:
            file_size  = len(changeFile.source_code)
        elif changeFile.source_code_before != None:
            file_size  = len(changeFile.source_code_before)
        self._add_feature('file_size', file_size)

            
    def _mine_code_analysis(self, changeFile: Modification):
        if not changeFile.language_supported:
            return

        changed_methods_count = len(changeFile.changed_methods)
        self._add_feature('changed_methods_count', changed_methods_count)
        total_methods_count = len(changeFile.methods)
        self._add_feature('total_methods_count', total_methods_count)
        file_complexity = changeFile.complexity if changeFile.complexity != None else 0
        self._add_feature('file_complexity', file_complexity)
        file_nloc = changeFile.nloc if changeFile.nloc != None else 0
        self._add_feature('file_nloc', file_nloc)
        file_token_count = changeFile.token_count if changeFile.token_count != None else 0
        self._add_feature('file_token_count', file_token_count)

        max_method_token_count = 0
        avg_method_token_count = 0
        max_method_complexity = 0
        avg_method_complexity = 0
        max_method_nloc = 0
        avg_method_nloc = 0
        max_method_parameter_count = 0
        avg_method_parameter_count = 0
        keywords_in_method_names = {k:0 for k in constants_and_configs.better_security_keywords}

        for method in changeFile.changed_methods:
            method_name_lower = method.long_name.lower()
            max_method_token_count = max(max_method_token_count, method.token_count)
            avg_method_token_count = avg_method_token_count + method.token_count

            max_method_complexity = max(max_method_complexity, method.complexity)
            avg_method_complexity = avg_method_complexity + method.complexity

            max_method_nloc = max(max_method_nloc, method.nloc)
            avg_method_nloc = avg_method_nloc + method.nloc

            max_method_parameter_count = max(max_method_parameter_count, len(method.parameters))
            avg_method_parameter_count = avg_method_parameter_count + len(method.parameters)

            for k in constants_and_configs.better_security_keywords:
                if k in method_name_lower:
                    keywords_in_method_names[k] +=1

        self._add_feature('max_method_token_count', max_method_token_count)
        self._add_feature('max_method_complexity', max_method_complexity)
        self._add_feature('max_method_nloc', max_method_nloc)
        self._add_feature('max_method_parameter_count', max_method_parameter_count)

        save_file_changed_method_count = changed_methods_count if changed_methods_count > 0 else 1
        self._add_feature('avg_method_token_count', avg_method_token_count/save_file_changed_method_count)
        self._add_feature('avg_method_complexity', avg_method_complexity/save_file_changed_method_count)
        self._add_feature('avg_method_nloc', avg_method_nloc/save_file_changed_method_count)
        self._add_feature('avg_method_parameter_count', avg_method_parameter_count/save_file_changed_method_count)

        for k, v in keywords_in_method_names.items():
            self._add_feature(f'methods_with_{k}_count', v)

         
    def _mine_bag_of_words(self,changeFile: Modification):
        filename = changeFile.filename.lower()
        path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
        path = path.lower()

        test_in_filename = 'test' in filename
        self._add_feature('test_in_filename', test_in_filename)
        test_in_path = 'test' in path
        self._add_feature('test_in_path', test_in_path)

        diff_scaped = self._escape_separators(changeFile.diff.lower())

        file_content = ''
        if changeFile.source_code != None:
            file_content  = self._escape_separators(changeFile.source_code.lower())
        elif changeFile.source_code_before != None:
            file_content  = self._escape_separators(changeFile.source_code_before.lower())

        for k in constants_and_configs.better_security_keywords:
            if k in diff_scaped:
                self._add_feature(f'{k}_in_patch', True)
            else:
                self._add_feature(f'{k}_in_patch', False)
            if k in file_content:
                self._add_feature(f'{k}_in_file_content', True)
            else:
                self._add_feature(f'{k}_in_file_content', False)


    def _mine_history_based(self, changeFile, history, index):
        safe_path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path

        changes_to_file_in_prev_50_commits = 0
        is_file_recently_added = False
        for offset in range(-50, -1):
            i = index + offset
            if i < 0:
                continue
            if i >= len(history): 
                continue

            for mod in history[i].modifications:
                if safe_path == mod.new_path or safe_path == mod.old_path:
                    changes_to_file_in_prev_50_commits += 1
                    if mod.change_type == ModificationType.ADD:
                        is_file_recently_added = True

        self._add_feature('changes_to_file_in_prev_50_commits', changes_to_file_in_prev_50_commits)
        self._add_feature('is_file_recently_added', is_file_recently_added)

        changes_to_file_in_next_50_commits = 0
        is_file_recently_removed = False
        for offset in range(1, 50):
            i = index + offset
            if i < 0:
                continue
            if i >= len(history): 
                continue

            for mod in history[i].modifications:
                if safe_path == mod.new_path or safe_path == mod.old_path:
                    changes_to_file_in_next_50_commits += 1
                    if mod.change_type == ModificationType.DELETE:
                        is_file_recently_removed = True


        self._add_feature('changes_to_file_in_next_50_commits', changes_to_file_in_next_50_commits)
        self._add_feature('is_file_recently_removed', is_file_recently_removed)
        
from pydriller import Modification,ModificationType
from os import path
from src.linked_commit_investigation import constants_and_configs
from src.linked_commit_investigation.AbstractMiner import AbstractMiner


class FileLevelFeatureMiner(AbstractMiner):
    def __init__(self, feature_classes) -> None:
        super().__init__(feature_classes)


    def mine_commit(self, changeFile: Modification):
        self.features = {}
        self._mine_labels(changeFile)
        if 'basic' in self.feature_classes:
            self._mine_basic(changeFile)
        if 'code_analysis' in self.feature_classes:
            self._mine_code_analysis(changeFile)
        if 'bag_of_words' in self.feature_classes:
            self._mine_bag_of_words(changeFile)
        return self.features


    def _mine_labels(self, changeFile: Modification):
        path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
        self._add_feature('label_filepath', path.lower())


    def _mine_basic(self, changeFile: Modification):
        filename = changeFile.filename.lower()
        _, file_extension = path.splitext(filename)
        
        is_code = (file_extension not in constants_and_configs.extensions_to_ignore) or (changeFile.language_supported)
        self._add_feature('is_code', is_code)
        self._add_feature('extension', file_extension)

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
        file_changed_method_count = len(changeFile.changed_methods)
        self._add_feature('file_changed_method_count', file_changed_method_count)

        max_method_token_count = 0
        avg_method_token_count = 0
        max_method_complexity = 0
        avg_method_complexity = 0
        max_method_nloc = 0
        avg_method_nloc = 0
        max_method_parameter_count = 0
        avg_method_parameter_count = 0
        has_methods_with_security_keywords = 0
        for method in changeFile.changed_methods:
            max_method_token_count = max(max_method_token_count, method.token_count)
            avg_method_token_count = avg_method_token_count + method.token_count

            max_method_complexity = max(max_method_complexity, method.complexity)
            avg_method_complexity = avg_method_complexity + method.complexity

            max_method_nloc = max(max_method_nloc, method.nloc)
            avg_method_nloc = avg_method_nloc + method.nloc

            max_method_parameter_count = max(max_method_parameter_count, len(method.parameters))
            avg_method_parameter_count = avg_method_parameter_count + len(method.parameters)

            if has_methods_with_security_keywords == 0:
                if any([True for x in constants_and_configs.security_keywords if x in method.long_name.lower()]):
                    has_methods_with_security_keywords = 1


        self._add_feature('max_method_token_count', max_method_token_count)
        self._add_feature('max_method_complexity', max_method_complexity)
        self._add_feature('max_method_nloc', max_method_nloc)
        self._add_feature('max_method_parameter_count', max_method_parameter_count)

        save_file_changed_method_count = file_changed_method_count if file_changed_method_count > 0 else 1
        self._add_feature('avg_method_token_count', avg_method_token_count/save_file_changed_method_count)
        self._add_feature('avg_method_complexity', avg_method_complexity/save_file_changed_method_count)
        self._add_feature('avg_method_nloc', avg_method_nloc/save_file_changed_method_count)
        self._add_feature('avg_method_parameter_count', avg_method_parameter_count/save_file_changed_method_count)

        self._add_feature('has_methods_with_security_keywords', has_methods_with_security_keywords)

         
    def _mine_bag_of_words(self,changeFile: Modification):
        filename = changeFile.filename.lower()
        path = changeFile.new_path if changeFile.new_path != None else changeFile.old_path
        path = path.lower()

        test_in_filename = 'test' in filename
        self._add_feature('test_in_filename', test_in_filename)
        test_in_path = 'test' in path
        self._add_feature('test_in_path', test_in_path)

        full_text = ''
        if changeFile.new_path != None:
            full_text += changeFile.new_path.lower()
        if changeFile.old_path != None and changeFile.old_path != changeFile.new_path:
            full_text += changeFile.old_path.lower()
        source = changeFile.source_code if changeFile.source_code != None else changeFile.source_code_before
        full_text += source

        full_text = self._escape_separators(full_text.lower())
        diff = self._escape_separators(changeFile.diff.lower())

        change_contains_cwe_title = any([True for x in constants_and_configs.cwe_titles if x in diff])
        self._add_feature('change_contains_cwe_title', change_contains_cwe_title)
        file_contains_cwe_title = any([True for x in constants_and_configs.cwe_titles if x in full_text])
        self._add_feature('file_contains_cwe_title', file_contains_cwe_title)
        change_contains_security_keyword = any([True for x in constants_and_configs.security_keywords if x in diff])
        self._add_feature('change_contains_security_keyword', change_contains_security_keyword)
        file_contains_security_keyword =  any([True for x in constants_and_configs.security_keywords if x in full_text])
        self._add_feature('file_contains_security_keyword', file_contains_security_keyword)

        


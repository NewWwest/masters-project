import pandas as pd

from src.linked_commit_investigation import constants_and_configs 

features_filterred_cache_location = 'results\checkpoints2'
features_calculated_cache_location = 'results\checkpoints3'


class FeatureCalculator:
    def __init__(self) -> None:
        pass

    def process_repo(self, repo_full_name):
        segments = repo_full_name.split('/')
        commits_df = pd.read_csv(f'{features_filterred_cache_location}/filtered-commit-level-{segments[0]}-{segments[1]}.csv')
        files_df = pd.read_csv(f'{features_filterred_cache_location}/filtered-file-level-{segments[0]}-{segments[1]}.csv')

        
        files = files_df.groupby('label_sha')
        features = []
        for commit_sha, files_for_commit in files:
            commit_info = commits_df[commits_df['label_sha'] == commit_sha]
            final_features = self.select_features_for_commit(commit_info.iloc[0], files_for_commit)
            features.append(final_features)

        features_df = pd.DataFrame(features)
        features_df.to_csv(f'{features_calculated_cache_location}/features-{segments[0]}-{segments[1]}.csv', index=False)



    def select_features_for_commit(self, commit_level, file_level):
        new_features = {}
        for f in features_to_copy_from_commit:
            if f not in commit_level:
                new_features[f] = float('nan')
            else:
                new_features[f]=commit_level[f]


        self.encode_extension(file_level, new_features)

        for f in features_to_agregate_as_flags:
            if f not in file_level or file_level.shape[0] == 0:
                new_features[f] = float('nan')
            else:
                ff = file_level[file_level[f]==1]
                new_features[f] = ff.shape[0]/file_level.shape[0]

        for f in features_to_agregate_as_avg_max_values:
            if f not in file_level or file_level.shape[0] == 0:
                new_features[f'{f}_avg'] = float('nan')
                new_features[f'{f}_max'] = float('nan')
            else:
                new_features[f'{f}_avg'] = file_level[f].mean()
                new_features[f'{f}_max'] = file_level[f].max()

        return new_features


    def encode_extension(self, files, feature_store):
        if 'extension' not in files:
            for ecosystem in constants_and_configs.code_extensions:
                feature_store[f'has_{ecosystem}'] = 0
            return

        extensions = set(files['extension'])
        for ecosystem in constants_and_configs.code_extensions:
            has_eco = any([True for x in constants_and_configs.code_extensions[ecosystem] if x in extensions])
            feature_store[f'has_{ecosystem}'] = 1 if has_eco else 0
        


features_to_copy_from_commit = [
    'label_repo_full_name',
    'label_sha',
    'label_commit_date',
    'author_to_commiter_date_diff',
    'same_author_as_commiter',
    'committed_by_bot',
    'authored_by_bot',
    'changed_files',
    'dmm_unit_complexity',
    'dmm_unit_interfacing',
    'dmm_unit_size',
    'message_contains_cwe_title',
    'title_contains_cwe_title',
    'message_contains_security_keyword',
    'title_contains_security_keyword',
]

features_to_agregate_as_flags = [
    'is_code',
    'is_add',
    'is_rename',
    'is_delete',
    'is_modify',
    'test_in_filename',
    'test_in_path',
    'has_methods_with_security_keywords',
    'change_contains_cwe_title',
    'file_contains_cwe_title',
    'change_contains_security_keyword',
    'file_contains_security_keyword',
]

features_to_agregate_as_avg_max_values = [
    'removed_lines_count',
    'added_lines_count',
    'changed_lines_count',
    'removed_lines_ratio',
    'added_lines_ratio',
    'modified_lines_count',
    'modified_lines_ratio',
    'file_size',
    'changed_methods_count',
    'total_methods_count',
    'file_complexity',
    'file_nloc',
    'file_token_count',
    'file_changed_method_count',
    'max_method_token_count',
    'max_method_complexity',
    'max_method_nloc',
    'max_method_parameter_count',
    'avg_method_token_count',
    'avg_method_complexity',
    'avg_method_nloc',
    'avg_method_parameter_count',
]



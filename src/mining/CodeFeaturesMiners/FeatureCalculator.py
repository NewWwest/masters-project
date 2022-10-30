import pandas as pd

import src.mining.CodeFeaturesMiners.constants_and_configs as constants_and_configs

features_cache_location = 'results/checkpoints3'

class FeatureCalculator:
    def __init__(self) -> None:
        pass

    def process_repo(self, commits_df, files_df, repo_full_name):
        segments = repo_full_name.split('/')

        if commits_df.shape[0] == 0 or files_df.shape[0] == 0:
            features_df = pd.DataFrame([])
            features_df.to_csv(f'{features_cache_location}/features-{segments[0]}-{segments[1]}.csv', index=False)
            return
        
        files = files_df.groupby('label_sha')
        features = []
        for commit_sha, files_for_commit in files:
            commit_info = commits_df[commits_df['label_sha'] == commit_sha]
            final_features = self.select_features_for_commit(commit_info.iloc[0], files_for_commit)
            features.append(final_features)

        features_df = pd.DataFrame(features)
        features_df.to_csv(f'{features_cache_location}/features-{segments[0]}-{segments[1]}.csv', index=False)
        return features_df



    def select_features_for_commit(self, commit_level, file_level):
        new_features = {}
        for f in features_to_copy_from_commit:
            if f in commit_level:
                new_features[f] = commit_level[f]
            else:
                new_features[f] = float('nan')

        for f in features_to_agregate_as_flags:
            if file_level.shape[0] == 0:
                new_features[f] = float('nan')
            else:
                if f in file_level:
                    ff = file_level[file_level[f]==1]
                    new_features[f] = ff.shape[0]/file_level.shape[0]
                else:
                    new_features[f] = float('nan')

        for f in features_to_agregate_as_avg_max_values:
            if file_level.shape[0] == 0:
                new_features[f'{f}_avg'] = float('nan')
                new_features[f'{f}_max'] = float('nan')
            else:
                if f in file_level:
                    new_features[f'{f}_avg'] = file_level[f].mean()
                    new_features[f'{f}_max'] = file_level[f].max()
                else:
                    new_features[f'{f}_avg'] = float('nan')
                    new_features[f'{f}_max'] = float('nan')


        new_features['changed_files'] = len(file_level)

        return new_features



features_to_copy_from_commit = [
    'label_repo_full_name',
    'label_sha',
    'label_commit_date',
    'author_to_commiter_date_diff',
    'same_author_as_commiter',
    'committed_by_bot',
    'authored_by_bot',
    'author_in_top_100',
    'dmm_unit_complexity',
    'dmm_unit_interfacing',
    'dmm_unit_size',

    'secur_in_title',
    'secur_in_message',
    'vulnerab_in_title',
    'exploit_in_title',
    'vulnerab_in_message',
    'exploit_in_message',
    'certificat_in_title',
    'certificat_in_message',
    'authent_in_title',
    'authent_in_message',
    'leak_in_title',
    'leak_in_message',
    'sensit_in_title',
    'sensit_in_message',
    'crash_in_title',
    'crash_in_message',
    'attack_in_title',
    'attack_in_message',
    'deadlock_in_title',
    'deadlock_in_message',
    'segfault_in_title',
    'segfault_in_message',
    'malicious_in_title',
    'malicious_in_message',
    'corrupt_in_title',
    'corrupt_in_message',

    'commits_prev_7_days',
    'commits_next_7_days',
    'commits_next_30_days',
    'time_to_next_merge',
    'commits_to_next_merge',
    'commits_since_last_merge',
    'time_to_prev_commit',
    'time_to_next_commit'
]





features_to_agregate_as_flags = [
    'has_npm_code',
    'has_npm_like_code',
    'has_mvn_code',
    'has_mvn_like_code',
    'has_pypi_code',
    'has_pypi_like_code',

    'is_add',
    'is_rename',
    'is_delete',
    'is_modify',

    'test_in_filename',
    'test_in_path',

    'is_file_recently_added',
    'is_file_recently_removed',

    'secur_in_file_content',
    'vulnerab_in_file_content',
    'exploit_in_file_content', 
    'certificat_in_file_content', 
    'authent_in_file_content',
    'leak_in_file_content',
    'sensit_in_file_content',
    'crash_in_file_content',
    'attack_in_file_content',
    'deadlock_in_file_content',
    'segfault_in_file_content',
    'malicious_in_file_content',
    'corrupt_in_file_content',

    'secur_in_patch',
    'vulnerab_in_patch',
    'exploit_in_patch', 
    'certificat_in_patch', 
    'authent_in_patch',
    'leak_in_patch',
    'sensit_in_patch',
    'crash_in_patch',
    'attack_in_patch',
    'deadlock_in_patch',
    'segfault_in_patch',
    'malicious_in_patch',
    'corrupt_in_patch'
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
    'max_method_token_count',
    'max_method_complexity',
    'max_method_nloc',
    'max_method_parameter_count',
    'avg_method_token_count',
    'avg_method_complexity',
    'avg_method_nloc',
    'avg_method_parameter_count',

    'methods_with_secur_count',
    'methods_with_vulnerab_count',
    'methods_with_exploit_count',
    'methods_with_certificat_count',
    'methods_with_authent_count',
    'methods_with_leak_count',
    'methods_with_sensit_count',
    'methods_with_crash_count',
    'methods_with_attack_count',
    'methods_with_deadlock_count',
    'methods_with_segfault_count',
    'methods_with_malicious_count',
    'methods_with_corrupt_count',

    'changes_to_file_in_prev_50_commits',
    'changes_to_file_in_next_50_commits',
]



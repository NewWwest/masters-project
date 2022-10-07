import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
from src.rq4.CommitProvider import CommitProvider
from src.utils.utils import get_files_in_from_directory

security_commits_file = 'src/datasets/security_related_commits_in_vuln.csv'
features_location = 'results\checkpoints1'
features_location_extra = 'results\checkpoints3'
ecosystem_threshold = 0.14

data_files = get_files_in_from_directory(features_location, '.csv', startswith= 'features')
extra_data_files = get_files_in_from_directory(features_location_extra, '.csv', startswith= 'features')

def load_files(collection):
    dfs = []
    for f in collection:
        try:
            dfs.append(pd.read_csv(f))
        except:
            pass
    return pd.concat(dfs)

df_normal = load_files(data_files) 
df_extra = load_files(extra_data_files) 
df = pd.concat([df_normal, df_extra])
df.drop_duplicates(['label_repo_full_name', 'label_sha'], inplace=True)

commitProvider = CommitProvider(security_commits_file)
df['label_security_related'] = df.apply(lambda r: commitProvider.is_security_related(r['label_repo_full_name'], r['label_sha']), axis=1)

df.to_csv('features.csv', index=False)

df_npm = df[df.apply(lambda r: r['has_npm_code'] > ecosystem_threshold, axis=1)]
df_npm.to_csv('features_npm.csv', index=False)

df_pypi = df[df.apply(lambda r: r['has_pypi_code'] > ecosystem_threshold, axis=1)]
df_pypi.to_csv('features_pypi.csv', index=False)

df_mvn = df[df.apply(lambda r: r['has_mvn_code'] > ecosystem_threshold, axis=1)]
df_mvn.to_csv('features_mvn.csv', index=False)



print('df', (df['label_security_related']==True).sum(), df.shape)
print('df_npm', (df_npm['label_security_related']==True).sum(), df_npm.shape)
print('df_pypi', (df_pypi['label_security_related']==True).sum(), df_pypi.shape)
print('df_mvn', (df_mvn['label_security_related']==True).sum(), df_mvn.shape)
print('ok')

# label_repo_full_name,
# label_sha,
# label_commit_date,
# author_to_commiter_date_diff,
# same_author_as_commiter,
# committed_by_bot,
# authored_by_bot,
# author_in_top_100,
# dmm_unit_complexity,
# dmm_unit_interfacing,
# dmm_unit_size,
# secur_in_title,
# secur_in_message,
# vulnerab_in_title,
# exploit_in_title,
# vulnerab_in_message,
# exploit_in_message,
# certificat_in_title,
# certificat_in_message,
# authent_in_title,
# authent_in_message,
# leak_in_title,
# leak_in_message,
# sensit_in_title,
# sensit_in_message,
# crash_in_title,
# crash_in_message,
# attack_in_title,
# attack_in_message,
# deadlock_in_title,
# deadlock_in_message,
# segfault_in_title,
# segfault_in_message,
# malicious_in_title,
# malicious_in_message,
# corrupt_in_title,
# corrupt_in_message,
# commits_prev_7_days,
# commits_next_7_days,
# commits_next_30_days,
# time_to_next_merge,
# commits_to_next_merge,
# commits_since_last_merge,
# time_to_prev_commit,
# time_to_next_commit,
# has_npm_code,
# has_npm_like_code,
# has_mvn_code,
# has_mvn_like_code,
# has_pypi_code,
# has_pypi_like_code,
# is_add,
# is_rename,
# is_delete,
# is_modify,
# test_in_filename,
# test_in_path,
# is_file_recently_added,
# is_file_recently_removed,
# secur_in_file_content,
# vulnerab_in_file_content,
# exploit_in_file_content,
# certificat_in_file_content,
# authent_in_file_content,
# leak_in_file_content,
# sensit_in_file_content,
# crash_in_file_content,
# attack_in_file_content,
# deadlock_in_file_content,
# segfault_in_file_content,
# malicious_in_file_content,
# corrupt_in_file_content,
# secur_in_patch,
# vulnerab_in_patch,
# exploit_in_patch,
# certificat_in_patch,
# authent_in_patch,
# leak_in_patch,
# sensit_in_patch,
# crash_in_patch,
# attack_in_patch,
# deadlock_in_patch,
# segfault_in_patch,
# malicious_in_patch,
# corrupt_in_patch,
# removed_lines_count_avg,
# removed_lines_count_max,
# added_lines_count_avg,
# added_lines_count_max,
# changed_lines_count_avg,
# changed_lines_count_max,
# removed_lines_ratio_avg,
# removed_lines_ratio_max,
# added_lines_ratio_avg,
# added_lines_ratio_max,
# modified_lines_count_avg,
# modified_lines_count_max,
# modified_lines_ratio_avg,
# modified_lines_ratio_max,
# file_size_avg,
# file_size_max,
# changed_methods_count_avg,
# changed_methods_count_max,
# total_methods_count_avg,
# total_methods_count_max,
# file_complexity_avg,
# file_complexity_max,
# file_nloc_avg,
# file_nloc_max,
# file_token_count_avg,
# file_token_count_max,
# file_changed_method_count_avg,
# file_changed_method_count_max,
# max_method_token_count_avg,
# max_method_token_count_max,
# max_method_complexity_avg,
# max_method_complexity_max,
# max_method_nloc_avg,
# max_method_nloc_max,
# max_method_parameter_count_avg,
# max_method_parameter_count_max,
# avg_method_token_count_avg,
# avg_method_token_count_max,
# avg_method_complexity_avg,
# avg_method_complexity_max,
# avg_method_nloc_avg,
# avg_method_nloc_max,
# avg_method_parameter_count_avg,
# avg_method_parameter_count_max,
# methods_with_secur_count_avg,
# methods_with_secur_count_max,
# methods_with_vulnerab_count_avg,
# methods_with_vulnerab_count_max,
# methods_with_exploit_count_avg,
# methods_with_exploit_count_max,
# methods_with_certificat_count_avg,
# methods_with_certificat_count_max,
# methods_with_authent_count_avg,
# methods_with_authent_count_max,
# methods_with_leak_count_avg,
# methods_with_leak_count_max,
# methods_with_sensit_count_avg,
# methods_with_sensit_count_max,
# methods_with_crash_count_avg,
# methods_with_crash_count_max,
# methods_with_attack_count_avg,
# methods_with_attack_count_max,
# methods_with_deadlock_count_avg,
# methods_with_deadlock_count_max,
# methods_with_segfault_count_avg,
# methods_with_segfault_count_max,
# methods_with_malicious_count_avg,
# methods_with_malicious_count_max,
# methods_with_corrupt_count_avg,
# methods_with_corrupt_count_max,
# changes_to_file_in_prev_50_commits_avg,
# changes_to_file_in_prev_50_commits_max,
# changes_to_file_in_next_50_commits_avg,
# changes_to_file_in_next_50_commits_max,
# changed_files

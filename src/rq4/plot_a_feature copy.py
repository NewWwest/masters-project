import matplotlib.pyplot as plt
from dateutil import parser
import glob
import pandas as pd

path = 'results\checkpoints4'

def main():
    # single_repo = 'WordPress-WordPress'
    single_repo = None
    if single_repo != None:
        df = pd.read_csv(fr'D:\Projects\2022\results\checkpoints4\labelled-features-{single_repo}.csv')
    else:
        csv_files = glob.glob(path + "/*.csv")
        dfs = (pd.read_csv(file) for file in csv_files)
        df   = pd.concat(dfs, ignore_index=True)

    scatter_feature(df, feature='author_to_commiter_date_diff', ymax=800) 
    violin_feature(df, feature='author_to_commiter_date_diff', ymax=800) 
    # compare(df, feature_x='file_size_avg',feature_y='file_changed_method_count_avg', xmax=2000, ymax=50) 


def scatter_feature(df, feature=None, repo=None, ymax = None):
    if feature == None:
        return
    if repo != None:
        df = df[df['label_repo_full_name']==repo]

    df = df[df[feature].isna() == False]
    fixes = df[df['is_fix']] 
    not_fixes = df[df['is_fix'] != True] 

    X_n = [parser.parse(x) for x in not_fixes['label_commit_date']]
    X_p = [parser.parse(x) for x in fixes['label_commit_date']]
    plt.scatter(X_n, not_fixes[feature], c='c', marker='*', )
    plt.scatter(X_p, fixes[feature], c='r', marker='o')
    if ymax != None:
        plt.ylim(0, ymax)
    plt.show()

def violin_feature(df, feature=None, repo=None, ymax = None):
    if feature == None:
        return
    if repo != None:
        df = df[df['label_repo_full_name']==repo]

    df = df[df[feature].isna() == False]
    df = df[df[feature] == 0]
    fixes = df[df['is_fix']] 
    not_fixes = df[df['is_fix'] != True] 

    if ymax != None:
        plt.ylim(0, ymax)

    plt.violinplot([not_fixes[feature], fixes[feature]], showmeans=False, showextrema=False)
    plt.boxplot([not_fixes[feature], fixes[feature]])
    plt.show()

def compare(df, feature_x=None, feature_y=None, repo=None, ymax = None, xmax=None):
    if feature_x == None or feature_y == None:
        return
    if repo != None:
        df = df[df['label_repo_full_name']==repo]

    df = df[df[feature_x].isna() == False]
    df = df[df[feature_y].isna() == False]
    fixes = df[df['is_fix']] 
    not_fixes = df[df['is_fix'] != True] 

    if ymax != None:
        plt.ylim(0, ymax)
    if xmax != None:
        plt.xlim(0, xmax)

    plt.scatter(not_fixes[feature_x], not_fixes[feature_y], c='c', marker='*', )
    plt.scatter(fixes[feature_x], fixes[feature_y], c='r', marker='o')
    plt.show()




if __name__ == '__main__':
    main()





# label_repo_full_name
# label_sha
# label_commit_date
# is_fix

# author_to_commiter_date_diff
# same_author_as_commiter
# committed_by_bot
# authored_by_bot
# changed_files
# is_merge
# dmm_unit_complexity
# dmm_unit_interfacing
# dmm_unit_size
# message_contains_cwe_title
# title_contains_cwe_title
# message_contains_security_keyword
# title_contains_security_keyword
# has_npm
# has_mvn
# has_pypi
# is_code
# is_add
# is_rename
# is_delete
# is_modify
# test_in_filename
# test_in_path
# has_methods_with_security_keywords
# change_contains_cwe_title
# file_contains_cwe_title
# change_contains_security_keyword
# file_contains_security_keyword
# removed_lines_count_avg
# removed_lines_count_max
# added_lines_count_avg
# added_lines_count_max
# changed_lines_count_avg
# changed_lines_count_max
# removed_lines_ratio_avg
# removed_lines_ratio_max
# added_lines_ratio_avg
# added_lines_ratio_max
# modified_lines_count_avg
# modified_lines_count_max
# modified_lines_ratio_avg
# modified_lines_ratio_max
# file_size_avg
# file_size_max
# changed_methods_count_avg
# changed_methods_count_max
# total_methods_count_avg
# total_methods_count_max
# file_complexity_avg
# file_complexity_max
# file_nloc_avg
# file_nloc_max
# file_token_count_avg
# file_token_count_max
# file_changed_method_count_avg
# file_changed_method_count_max
# max_method_fan_in_avg TODO: remove
# max_method_fan_in_max TODO: remove
# max_method_fan_out_avg TODO: remove
# max_method_fan_out_max TODO: remove
# max_method_general_fan_out_avg TODO: remove
# max_method_general_fan_out_max TODO: remove
# max_method_token_count_avg
# max_method_token_count_max
# max_method_complexity_avg
# max_method_complexity_max
# max_method_nloc_avg
# max_method_nloc_max
# max_method_parameter_count_avg
# max_method_parameter_count_max
# avg_method_fan_in_avg  TODO: remove
# avg_method_fan_in_max  TODO: remove
# avg_method_fan_out_avg  TODO: remove
# avg_method_fan_out_max TODO: remove
# avg_method_general_fan_out_avg TODO: remove
# avg_method_general_fan_out_max TODO: remove
# avg_method_token_count_avg
# avg_method_token_count_max
# avg_method_complexity_avg
# avg_method_complexity_max
# avg_method_nloc_avg
# avg_method_nloc_max
# avg_method_parameter_count_avg
# avg_method_parameter_count_max

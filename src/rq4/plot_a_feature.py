import matplotlib.pyplot as plt
import pandas as pd

big_boi = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projects.csv'
gh_boi = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projectsx.csv'


def plot_feature_vs_time(df, feature=None, repo=None, ymax = None):
    if feature == None:
        return
    if repo != None:
        df = df[df['repo_full_name']==repo]

    df = df[df[feature].isnull() == False]
    fixes = df[df['is_vulnerability_fix']] 
    not_fixes = df[df['is_vulnerability_fix'] != True] 

    plt.scatter(not_fixes['commit_days_ago'], not_fixes[feature], c='c', marker='*', )
    plt.scatter(fixes['commit_days_ago'], fixes[feature], c='r', marker='o')
    if ymax != None:
        plt.ylim(0, ymax)
    plt.show()

def plot_box_plot(df, feature=None, repo=None, ymax = None):
    if feature == None:
        return
    if repo != None:
        df = df[df['repo_full_name']==repo]

    df = df[df[feature].isnull() == False]
    fixes = df[df['is_vulnerability_fix']] 
    not_fixes = df[df['is_vulnerability_fix'] != True] 

    if ymax != None:
        plt.ylim(0, ymax)

    plt.boxplot([not_fixes[feature], fixes[feature]])
    # plt.boxplot([[1 if x else 0 for x in not_fixes[feature]], [1 if x else 0 for x in fixes[feature]]])
    plt.show()

df = pd.read_csv(gh_boi)
plot_feature_vs_time(df, feature='hours_diff') 






# todo: biggest file in commit


# delta metric -> fixes are more binary
# avg_complexity -> looks useless
# avg_nloc -> looks useless
# avg_tokens -> looks useless
# hours diff -> fixes are merged very fast
# change-size -> fixes are small
# change-size -> fixes add more than they remove

# informative:
# df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_commits-moment-nocode.csv')
# plot_feature_vs_time(df, feature='hours_diff',repo='moment/moment', ymax=4000)
# plot_feature_vs_time(df, feature='changed_files',repo='moment/moment')
# plot_feature_vs_time(df, feature='avg_added_count',repo='moment/moment', ymax=2000)
# plot_feature_vs_time(df, feature='avg_file_size',repo='moment/moment', ymax=300_000) # very large files (?)
# plot_feature_vs_time(df, feature='contains_security_info',repo='moment/moment')
# plot_feature_vs_time(df, feature='contains_security_in_message',repo='moment/moment') 
 



# df = pd.read_csv('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_commits-tf-all.csv')
# plot_box_plot(df, feature='dmm_unit_complexity', repo='tensorflow/tensorflow')
# plot_box_plot(df, feature='dmm_unit_size', repo='tensorflow/tensorflow')
# plot_box_plot(df, feature='avg_removed_count', repo='tensorflow/tensorflow')
# plot_box_plot(df, feature='changed_files', ymax=20)
# plot_box_plot(df, feature='avg_mod_count')
# plot_box_plot(df, feature='avg_changed_methods')

# # all nloc/complexity/file_size/avg_tokens suggest the vulnerability fix happen in larger files
# plot_box_plot(df, feature='avg_nloc')
# plot_box_plot(df, feature='avg_complexity')
# plot_box_plot(df, feature='avg_file_size') 



# npm_repos = [ 'moment/moment','lodash/lodash','eslint/eslint','webpack/webpack-dev-server','caolan/async']
# pypi_repos = ['psf/requests', 'numpy/numpy','django/django', 'python-pillow/Pillow','scipy/scipy']
# mvn_repos = ['junit-team/junit4','google/guava','spring-projects/spring-framework','h2database/h2database','google/gson']
# repos = npm_repos + pypi_repos + mvn_repos
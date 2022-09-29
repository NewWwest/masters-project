from CommitMiner import CommitMiner
from RepoDownloader import RepoDownloader
import os
import pandas as pd

repositories_path = '/Users/awestfalewicz/Private/data/repositories'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mapping_vulnerabilities_to_commits.json'
mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec-star.json'
# mapping_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/sec.json'
checkpoint_path = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/results/fix_commit_checkpoints'
all_commits_count = 10000

ouput_file = 'mined_features_for_issues_most_starred.csv'
ouput_file_just_fixes = 'mined_features_for_most_common_projectsz-fixes.csv'

def mine_most_common_projects(skip_linux=True):
    cm = CommitMiner(repositories_path)
    rd = RepoDownloader(repositories_path)
    cm.load_mapping_file(mapping_file)

    with_fix_count = []
    for x in cm.marked_commits_repo:
        with_fix_count.append((x, len(cm.marked_commits_repo[x])))

    with_fix_count = sorted(with_fix_count, key=lambda x: x[1], reverse=True)

    all_dfs = []
    fix_dfs = []
    all_count = 0
    start = 1 if skip_linux else 0 #skip linux repo.... tooooo big
    for x in range(start, len(with_fix_count)):
        try:
            repo_full_name = with_fix_count[x][0]
            rd.download_repos([repo_full_name])
            df_fix = cm.mine_fixes_from_repos([repo_full_name])
            df_norm = cm.sample_repos([repo_full_name], 10*df_fix.shape[0])
            df = cm.concat_result_dfs([df_fix, df_norm])
            
            checkpoint_file = repo_full_name.replace('/','-')
            df.to_csv(f'{checkpoint_path}/{checkpoint_file}.csv', index=False)

            all_dfs.append(df)
            fix_dfs.append(df_fix)
            all_count += df.shape[0]
            if all_count >= all_commits_count:
                break
        except Exception as e:
            print('Failed to process repo', repo_full_name)
            print(e)


    all_df = pd.concat(all_dfs)
    all_df.to_csv(ouput_file, index=False)
    all_fix_dfs = pd.concat(fix_dfs)
    all_fix_dfs.to_csv(ouput_file_just_fixes, index=False)


def concat_checkpoit_csvs():
    dataframes = []
    for root, subdirs, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(f'{root}/{file}')
                    dataframes.append(df)
                except:
                    print('Failed to load file', file)
    all_df = pd.concat(dataframes)
    all_df.to_csv(ouput_file, index=False)
    return all_df

if __name__ == '__main__':
    concat_checkpoit_csvs()
    # mine_most_common_projects()
    pass
    
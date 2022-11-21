#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import pandas as pd
from src.utils.utils import get_files_in_from_directory

security_commits_file = 'data/security_related_commits_in_vuln.csv'
features_location = 'results/features1'
features_location_extra = None #'results/features2'
ecosystem_threshold = 0.14


def load_files(collection):
    dfs = []
    for f in collection:
        try:
            dfs.append(pd.read_csv(f))
        except:
            pass
    return pd.concat(dfs)

def main():
    data_files = get_files_in_from_directory(features_location, '.csv', startswith= 'features')
    df_normal = load_files(data_files) 

    if features_location_extra :
        extra_data_files = get_files_in_from_directory(features_location_extra, '.csv', startswith= 'features')
        df_extra = load_files(extra_data_files) 
        df = pd.concat([df_normal, df_extra])
    else:
        df = df_normal

    df.drop_duplicates(['label_repo_full_name', 'label_sha'], inplace=True)

    mapping_csv = pd.read_csv(security_commits_file)
    commits = set()
    for i,r in mapping_csv.iterrows():
        commits.add(f'{r["repo_owner"]}/{r["repo_name"]}/{r["commit_sha"]}')

    df['label_security_related'] = df.apply(lambda r: f'{r["label_repo_full_name"]}/{r["label_sha"]}' in commits, axis=1)

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


if __name__ == '__main__':
    main()

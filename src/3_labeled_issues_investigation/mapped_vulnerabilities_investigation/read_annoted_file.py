#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json
import pandas as pd
import time
from src.utils.utils import get_files_in_from_directory
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
from src.proxies.GitHubProxy import GithubProxy


input_issues_references_dataset = [
    r'data\most_starred\manually_annotated_mapping.csv',
    r'data\most_used_npm\manually_annotated_mapping.csv',
    r'data\most_used_pypi\manually_annotated_mapping.csv',
    r'data\most_used_mvn\manually_annotated_mapping.csv',
]

def main():
    issues_df = pd.concat([pd.read_csv(x) for x in input_issues_references_dataset])

    not_nan = 0
    anan = 0
    for i, r in issues_df.iterrows():
        if r['cve_id'].strip() != 'nan':
            not_nan +=1
        else:
            anan += 1
            
    print(not_nan)
    print(anan)
    print(issues_df.shape)
    print(not_nan/issues_df.shape[0])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

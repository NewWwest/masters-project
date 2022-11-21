import sys
sys.path.insert(0, r'D:\Projects\aaa')


from statistics import mean, median
from collections import Counter
import regex as re
from src.utils.utils import get_files_in_from_directory
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
import json
import dateutil.parser as parser
import pandas as pd
import matplotlib.pyplot as plt
import random

commits = set()
commits_data = {}
input_data_location = 'results/checkpoints_fixMapper'
classified_commits_file = r'src\rq4\classificated_commits.json'

with open(classified_commits_file, 'r') as f:
    classified_commits = json.load(f)

def was_classified(repo_full_name, sha):
    global classified_commits
    return any([True for x in classified_commits if x['label_sha']==sha and x['label_repo_full_name']==repo_full_name])

def was_classified_and_as_sec_relevant(repo_full_name, sha):
    global classified_commits
    return any([True for x in classified_commits if x['label_sha']==sha and x['label_repo_full_name']==repo_full_name and x['classification_pred']])


def add_date(ref_aliases, date, by_cve):
    for alias_id in ref_aliases:
        if alias_id not in by_cve:
            by_cve[alias_id] = []
        by_cve[alias_id].append(date)


def dates_from_method():
    data_files = get_files_in_from_directory(input_data_location, '.json')
    classified_issues = {}
    security_issues = {}

    for data_file in data_files:
        print(f'Processing... {data_file}')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for report_id in data:
            for ref in data[report_id]:
                if ref['reference_type'] == 'commit':
                    asd = json.loads(ref['data'])
                    repo_full_name = f"{ref['repo_owner']}/{ref['repo_name']}"
                    sha = ref['reference_value']
                    if was_classified(repo_full_name, sha):
                        commiter_date = parser.parse(asd['commit']['committer']['date'])
                        ref_aliases = json.loads(ref['aliases'])
                        add_date(ref_aliases, commiter_date, classified_issues)

                    if was_classified_and_as_sec_relevant(repo_full_name, sha):
                        commiter_date = parser.parse(asd['commit']['committer']['date'])
                        ref_aliases = json.loads(ref['aliases'])
                        add_date(ref_aliases, commiter_date, security_issues)

    return classified_issues, security_issues 
    


# Calculates the percentage of vulnerabilities that has at least one security related commit classified
def main(ecosystem = None):
    classified_issues, security_issues = dates_from_method()

    print(len(security_issues.keys())/len(classified_issues.keys()))
    print(ecosystem)


if __name__ == '__main__':
    # main('npm')  #0.341
    # main('pypi') #0.292
    # main('maven') #0.250
    main()
    print('ok')


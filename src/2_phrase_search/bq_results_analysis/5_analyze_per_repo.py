#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import pandas as pd
import csv
import json

input_file = r'data\results_from_bq_search.csv'
all_issues_input_file = r'all_issues_scanned.csv'
repos_file = r'data/repos/most_popular_pypi_packages.csv'
repos_datasets = [
    'src/rq3/repos/top_starred_repositories.csv',
    'src/rq3/repos/most_popular_maven_packages.csv',
    'src/rq3/repos/most_popular_npm_packages.csv',
    'src/rq3/repos/most_popular_pypi_packages.csv'
]


def count_processed_objects_per_repo():
    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)

    all_objects = set()
    with open(all_issues_input_file, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        i = 0
        for row in csv_reader:
            i += 1
            if i == 1:
                print(row)
                continue

            if i % 100_000 == 0:
                print(i)

            if row[0]:
                all_objects.add(row[0])
            elif row[1]:
                all_objects.add(row[1])

    by_repo = {}
    i = 0
    for x in all_objects:
        i += 1
        if i % 100_000 == 0:
            print(x)

        segments = x.split("/")
        repo_full_name= f'{segments[4]}/{segments[5]}'
        if repo_full_name not in by_repo:
            by_repo[repo_full_name] = 1
        else:
            by_repo[repo_full_name] += 1

    with open(r'src\rq3\bq_results_analysis\results\counts_per_repo.json', 'w') as f:
        json.dump(by_repo, f, indent=2)


# count_processed_objects_per_repo()
with open(r'src\rq3\bq_results_analysis\results\counts_per_repo.json', 'r') as f:
    all_object_count_per_repo = json.load(f)



df = pd.read_csv(input_file)
df['main_entity_url'] = df.apply(lambda x: x['commit_url'] if pd.isna(x['issue_url']) else x['issue_url'], axis=1)
df['repo_full_name'] = df.apply(lambda x: f'{x["main_entity_url"].split("/")[4]}/{x["main_entity_url"].split("/")[5]}', axis=1)
by_repo = df.groupby('repo_full_name')
object_count_per_repo = {k: len(v) for k, v in sorted(by_repo, key=lambda item: -len(item[1]))}

data_new = []
not_scanned = []
no_keywords = []
fractions = []
all_objects_scanned_sum = 0 
with_keywords_sum = 0 

repos = pd.read_csv(repos_file)
for i, r in repos.iterrows():
    repo_full_name = r['full_name']
    print(repo_full_name)

    if repo_full_name in all_object_count_per_repo:
        scanned = all_object_count_per_repo[repo_full_name]
    else:
        scanned = 0
        not_scanned.append(repo_full_name)

    if repo_full_name in object_count_per_repo:
        with_keywords = object_count_per_repo[repo_full_name]
    else:
        with_keywords = 0
        no_keywords.append(repo_full_name)

    all_objects_scanned_sum += scanned
    with_keywords_sum += with_keywords

    if with_keywords > 0 and scanned > 0:
        fractions.append({
            'repo_full_name': repo_full_name,
            'fraction': with_keywords/scanned,
            'with_keywords': with_keywords,
            'scanned': scanned,
        })

fractions = pd.DataFrame(fractions)
fractions.sort_values('fraction', inplace=True)

print('all_objects_scanned_sum', all_objects_scanned_sum) #41894769
print('with_keywords_sum', with_keywords_sum) #2237211

with open('repos_with_no_keywords.json', 'w') as f:
    json.dump(no_keywords, f, indent=2)
with open('repos_with_no_objects.json', 'w') as f:
    json.dump(not_scanned, f, indent=2)

fractions.to_csv('repos_with_percantages.csv', index=False)

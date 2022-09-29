import sys
sys.path.insert(0, r'D:\Projects\aaa')

import random
import csv
import json

from src.utils.utils import get_files_in_from_directory

repositories_to_process = 'src/datasets/repos_to_process.txt'
partial_results_directory = 'results/keyword_code_search_gc'
result_file = 'results/keyword_code_search_gc_finals/run1.json'

partitions_output = 'src/rq3/code_search/cloud_compute/CodeScanning'
partitions = 8

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def main():
    with open(repositories_to_process, 'r') as f:
        repos = f.readlines()
    repos = list([r.strip() for r in repos])
    partial_results = get_files_in_from_directory(partial_results_directory, '.json')
    print('Loaded repos...')

    repos2 = []
    done_repos = []
    for repo in repos:
        segments = repo.split('/')
        filename = f'keywords_in_diff-{segments[0]}-{segments[1]}-final.json'
        if any([x for x in partial_results if x.endswith(filename)]):
           done_repos.append(repo)
        else:
           repos2.append(repo)
    print('Filtered done repos...')

    jsons = []
    for repo in done_repos:
        segments = repo.split('/')
        filename = f'keywords_in_diff-{segments[0]}-{segments[1]}-'
        for x in partial_results:
            if filename in x:
                with open(x, 'r') as f:
                    jsons.append(f.read())
    print('Loaded partial results...')

    with open(result_file, 'w') as f:
        json.dump(jsons, f)
    print('Created current result...')

    random.shuffle(repos2)
    data = list(split(repos2, partitions))
    for partition in range(partitions):
        with open(f'{partitions_output}/repos_to_process_{partition}.txt', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for x in data[partition]:
                spamwriter.writerow([x])
    print('Created partitions...')

    


if __name__ == '__main__':
    main()
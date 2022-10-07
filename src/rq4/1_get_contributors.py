import sys
sys.path.insert(0, r'D:\Projects\aaa')

from src.rq4.CommitProvider import CommitProvider

import pandas as pd
import json
import time
import requests
from datetime import datetime
from urllib.parse import urlparse
from urllib.parse import parse_qs

from src.proxies.GitHubProxy import GithubProxy

security_commits_file = 'results/security_related_commits_in_vuln.csv'

'https://api.github.com/repositories/106310/commits?per_page=1&since=2013-06-30T22%3A39%3A06.156547&page=906'
def main():
    result = {}
    commitProvider = CommitProvider(security_commits_file)
    repos = commitProvider.get_repos_with_at_least_n_commits(5)
    ghproxy = GithubProxy()
    for r, _ in repos:
        print(r)
        segments = r.split('/')
        resp = ghproxy.get_top_contributors(segments[0], segments[1])
        if resp:
            result[r] = resp.json()

    with open('contributors.json', 'w') as f:
        json.dump(result, f)

        


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
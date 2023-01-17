#!/usr/bin/env python3
#
# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------

# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json
import time
import pandas as pd

from src.proxies.GitHubProxy import GithubProxy

security_commits_file = 'data/security_related_commits_in_vuln.csv'
minimal_commits_count = 5
result_contributors_file = 'contributors.json'

def main():
    ghproxy = GithubProxy()
    
    mapping_csv = pd.read_csv(security_commits_file)
    commits = {}
    for i,r in mapping_csv.iterrows():
        repo_full_name = f'{r["repo_owner"]}/{r["repo_name"]}'
        if repo_full_name not in commits:
            commits[repo_full_name] = set()
        
        commits[repo_full_name].add(r['commit_sha'])

    with_fix_count = []
    for x in commits:
        if len(commits[x]) >= minimal_commits_count:
            with_fix_count.append((x, list(commits[x])))

    result = {}
    for r, _ in with_fix_count:
        print(r)
        segments = r.split('/')
        resp = ghproxy.get_top_contributors(segments[0], segments[1])
        if resp:
            result[r] = resp.json()

    with open(result_contributors_file, 'w') as f:
        json.dump(result, f)

        
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
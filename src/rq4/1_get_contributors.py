import sys
sys.path.insert(0, r'D:\Projects\aaa')

from src.rq4.CommitProvider import CommitProvider
import json
import time

from src.proxies.GitHubProxy import GithubProxy

security_commits_file = 'results/security_related_commits_in_vuln.csv'

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
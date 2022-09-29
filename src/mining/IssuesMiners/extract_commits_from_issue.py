def extract_commits_from_issue(repo_owner, repo_name, issue_data):
    repo_full_name = f'{repo_owner}/{repo_name}'
    commits = []

    commits += _extract_commits_from_timeline(repo_owner, repo_name, issue_data['timeline_data'])
    if 'pull_request' in issue_data:
        if 'pull_request_commits' in issue_data:
            pr_commits = issue_data['pull_request_commits']
            commits += [x['sha'] for x in pr_commits]
    else:
        cross_referenced_issues = [x['source']['issue'] for x in issue_data['timeline_data'] if x['event']=='cross-referenced']
        cross_referenced_events_in_the_same_repo = [x for x in cross_referenced_issues if x['repository']['full_name']==repo_full_name]
        for referenced_issue in cross_referenced_events_in_the_same_repo:
            try:
                commits += _extract_commits_from_timeline(repo_owner, repo_name, referenced_issue['timeline_data'])
                if 'pull_request' in referenced_issue:
                    if 'pull_request_commits' in referenced_issue:
                        pr_commits = referenced_issue['pull_request_commits']
                        commits += [x['sha'] for x in pr_commits]
            except:
                print('Failed to process ref issue')
            
    return list(set(commits))


def _extract_commits_from_timeline(repo_owner, repo_name, timeline):
    try:
        result = []
        referenced_commits = [x for x in timeline if 'commit_id' in x and x['commit_id'] != None]
        for c in referenced_commits:
            if 'html_url' in c:
                seg = c['html_url'].split('/')
                ref_repo_owner = seg[3]
                ref_repo_name = seg[4]
            else:
                seg = c['commit_url'].split('/')
                ref_repo_owner = seg[4]
                ref_repo_name = seg[5]

            if repo_owner == ref_repo_owner and repo_name == ref_repo_name:
                result.append(c['commit_id'])
        return result
    except:
        return []

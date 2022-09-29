import json
import pandas as pd 

features_calculated_cache_location = 'results\checkpoints3'
features_selected_cache_location = 'results\checkpoints4'
fix_to_norm_ratio = 10
class CommitSelector:
    def __init__(self, mapping_file) -> None:
        with open(mapping_file, 'r') as f:
            self.fix_commits_by_cve = json.load(f)

        self.fix_commits_by_repo = self._group_mapping_by_repo(self.fix_commits_by_cve)

    def _group_mapping_by_repo(self, mapping_object):
        result = {}
        for ghsa in mapping_object:
            for commit in mapping_object[ghsa]:
                repo = f'{commit["owner"]}/{commit["repo"]}'
                if repo not in result:
                    result[repo] = []
                result[repo].append(commit)

        return result

    def get_repos_with_atleastn_fix_commits(self, n):
        with_fix_count = []
        for x in self.fix_commits_by_repo:
            with_fix_count.append((x, len(self.fix_commits_by_repo[x])))
        with_fix_count = sorted(with_fix_count, key=lambda x: x[1], reverse=True)
        sorted_repos = [x[0] for x in with_fix_count if x[1]>=n]
        return sorted_repos


    def select_commits_for_repo(self, repo_full_name):
        segments= repo_full_name.split('/')
        fix_commits = set([x['sha'] for x in self.fix_commits_by_repo[repo_full_name]])
        mined_commits = pd.read_csv(f'{features_calculated_cache_location}/features-{segments[0]}-{segments[1]}.csv')
        if 'Unnamed: 0' in mined_commits:
            mined_commits.drop('Unnamed: 0', inplace=True, axis=1)

        selected_commits = self._select_commits_for_repo(repo_full_name, mined_commits, fix_commits)
        selected_commits.to_csv(f'{features_selected_cache_location}/labelled-features-{segments[0]}-{segments[1]}.csv', index=False)


    def _select_commits_for_repo(self, repo_full_name, mined_commits, fix_commits):
        fix_datapoints = mined_commits[mined_commits.apply(lambda x: (x['label_sha'] in fix_commits) or (x['label_sha'][0:10] in fix_commits), axis=1)]
        no_fix_datapoints = mined_commits[mined_commits.apply(lambda x: (x['label_sha'] not in fix_commits) and  (x['label_sha'][0:10] not in fix_commits), axis=1)]
        no_fix_datapoints = no_fix_datapoints.sample(min(fix_datapoints.shape[0] * fix_to_norm_ratio, no_fix_datapoints.shape[0]))

        fix_datapoints = pd.DataFrame(fix_datapoints)
        no_fix_datapoints = pd.DataFrame(no_fix_datapoints)

        fix_datapoints['is_fix'] = True
        no_fix_datapoints['is_fix'] = False
        df = pd.concat([fix_datapoints, no_fix_datapoints])
        return df
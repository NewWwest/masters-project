import pandas as pd



features_cache_location = 'results\checkpoints1'
features_filterred_cache_location = 'results\checkpoints2'


class CommitPreselector:
    def filter_commits_for_repo(self, repo_full_name):
        segments = repo_full_name.split('/')
        commits_df = pd.read_csv(f'{features_cache_location}/commit-level-{segments[0]}-{segments[1]}.csv')
        files_df = pd.read_csv(f'{features_cache_location}/file-level-{segments[0]}-{segments[1]}.csv')

        shas_to_keep = self.filter_shas(commits_df, files_df)
        
        commits_filtered = commits_df[commits_df.apply(lambda x: x['label_sha'] in shas_to_keep, axis=1)]
        files_filtered = files_df[files_df.apply(lambda x: x['label_sha'] in shas_to_keep, axis=1)]

        commits_filtered.to_csv(f'{features_filterred_cache_location}/filtered-commit-level-{segments[0]}-{segments[1]}.csv')
        files_filtered.to_csv(f'{features_filterred_cache_location}/filtered-file-level-{segments[0]}-{segments[1]}.csv')


    def filter_shas(self, commits_df: pd.DataFrame, files_df: pd.DataFrame) -> list:
        files = files_df.groupby('label_sha')
        shas_to_keep = []
        for commit_sha, files_for_commit in files:
            commit_info = commits_df[commits_df['label_sha'] == commit_sha]
            keep = self.process_commit(commit_info.iloc[0], files_for_commit)
            if keep:
                shas_to_keep.append(commit_sha)

        return shas_to_keep


    def process_commit(self, commit_lvl_features, file_lvl_features_list) -> bool:
        # consider date filere on label_commit_date
        # consider authored_by_bot filter
        # if commit_lvl_features['is_merge']:
        #     return False

        if file_lvl_features_list.shape[0] == 0:
            return False
        
        has_code = any(file_lvl_features_list['is_code'])
        if not has_code:
            return False

        return True
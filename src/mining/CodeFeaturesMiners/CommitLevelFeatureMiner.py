from pydriller import Modification,ModificationType,Commit
from os import path
from src.linked_commit_investigation import constants_and_configs

from src.linked_commit_investigation.AbstractMiner import AbstractMiner

class CommitLevelFeatureMiner(AbstractMiner):
    def __init__(self, feature_classes) -> None:
        super().__init__(feature_classes)


    def mine(self, commit: Commit, history_before, history_after):
        self.features = {}
        self._mine_labels(commit)
        if 'basic' in self.feature_classes:
            self._mine_basic(commit)
        if 'code_analysis' in self.feature_classes:
            self._mine_code_analysis(commit)
        if 'bag_of_words' in self.feature_classes:
            self._mine_bag_of_words(commit)
        if 'history' in self.feature_classes:
            self._mine_history_based(commit, history_before, history_after)
        return self.features


    def _mine_labels(self, commit: Commit):
        self._add_feature('label_sha', commit.hash)
        self._add_feature('label_commit_date', commit.committer_date)


    def _mine_basic(self, commit: Commit):
        author_to_commiter_date_diff = (commit.committer_date - commit.author_date).total_seconds()//3600
        self._add_feature('author_to_commiter_date_diff', author_to_commiter_date_diff)
        same_author = commit.author.email.lower() == commit.committer.email.lower()
        self._add_feature('same_author_as_commiter', same_author)
        committed_by_bot = 'bot' in commit.committer.email.lower()
        self._add_feature('committed_by_bot', committed_by_bot)
        authored_by_bot = 'bot' in commit.author.email.lower()
        self._add_feature('authored_by_bot', authored_by_bot)

        self._add_feature('changed_files', len(commit.modifications))
            
        
    def _mine_code_analysis(self, commit: Commit):
        self._add_feature('dmm_unit_complexity', commit.dmm_unit_complexity)
        self._add_feature('dmm_unit_interfacing', commit.dmm_unit_interfacing)
        self._add_feature('dmm_unit_size', commit.dmm_unit_size)


    def _mine_bag_of_words(self,commit: Commit):
        message = commit.msg.lower()
        message_escaped = self._escape_separators(message)
        title_escaped = self._escape_separators(message.split('\n')[0])

        message_contains_cwe_title = any([True for x in constants_and_configs.cwe_titles if x in message_escaped])
        self._add_feature('message_contains_cwe_title', message_contains_cwe_title)
        title_contains_cwe_title = any([True for x in constants_and_configs.cwe_titles if x in title_escaped])
        self._add_feature('title_contains_cwe_title', title_contains_cwe_title)

        message_contains_security_keyword = any([True for x in constants_and_configs.security_keywords if x in message_escaped])
        self._add_feature('message_contains_security_keyword', message_contains_security_keyword)
        title_contains_security_keyword =  any([True for x in constants_and_configs.security_keywords if x in title_escaped])
        self._add_feature('title_contains_security_keyword', title_contains_security_keyword)


    def _mine_history_based(self, commit, history_before, history_after):
        three_days_in_seconds = 3600*24*3
        commits_in_last_3_days = [x for x in history_before if (commit.author_date - x['author_date']).total_seconds() < three_days_in_seconds]
        self._add_feature('commits_in_last_3_days', len(commits_in_last_3_days))
        self._add_feature('merges_in_last_3_days', len([1 for x in commits_in_last_3_days if x['merge']]))

        commits_in_following_3_days = [x for x in history_after if (x['author_date']-commit.author_date).total_seconds() < three_days_in_seconds]
        self._add_feature('commits_in_following_3_days', len(commits_in_following_3_days))
        self._add_feature('merges_in_following_3_days', len([1 for x in commits_in_following_3_days if x['merge']]))
        
        prev_commit = history_before[0] if len(history_before) > 0 else None
        next_commit = history_after[0] if len(history_after) > 0 else None

        followed_by_merge = next_commit['merge'] if next_commit != None else False
        time_to_prev_commit = (commit.author_date - prev_commit['author_date']).total_seconds()/3600 if prev_commit != None else float('nan')
        time_to_next_commit = (next_commit['author_date'] - commit.author_date).total_seconds()/3600 if next_commit != None else float('nan')
        self._add_feature('followed_by_merge', followed_by_merge)
        self._add_feature('time_to_prev_commit', time_to_prev_commit)
        self._add_feature('time_to_next_commit', time_to_next_commit)

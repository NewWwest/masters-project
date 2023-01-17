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
from pydriller import Commit
from Levenshtein import distance as levenshtein_distance

import src.mining.CodeFeaturesMiners.constants_and_configs as constants_and_configs

from src.mining.CodeFeaturesMiners.AbstractMiner import AbstractMiner


class CommitLevelFeatureMiner(AbstractMiner):
    def __init__(self, feature_classes) -> None:
        super().__init__(feature_classes)


    def mine(self, repo_full_name, commit: Commit, history, index, contributors=None):
        self.features = {}
        self._mine_labels(commit, repo_full_name)
        if 'basic' in self.feature_classes:
            self._mine_basic(commit, contributors)
        if 'code_analysis' in self.feature_classes:
            self._mine_code_analysis(commit)
        if 'bag_of_words' in self.feature_classes:
            self._mine_bag_of_words(commit)
        if 'history' in self.feature_classes and history != None and index != None:
            self._mine_history_based(history, index)
        return self.features


    def _mine_labels(self, commit: Commit, repo_full_name):
        self._add_feature('label_sha', commit.hash)
        self._add_feature('label_commit_date', commit.committer_date)
        self._add_feature('label_repo_full_name', repo_full_name)


    def _mine_basic(self, commit: Commit, contributors):
        author_to_commiter_date_diff = (commit.committer_date - commit.author_date).total_seconds()/3600
        author = commit.author.email.lower()
        commiter = commit.committer.email.lower()

        self._add_feature('author_to_commiter_date_diff', author_to_commiter_date_diff)
        self._add_feature('same_author_as_commiter', author == commiter)
        self._add_feature('committed_by_bot', 'bot' in commiter)
        self._add_feature('authored_by_bot', 'bot' in author)

        author_in_top_100 = False
        email_part = author.split('@')[0]
        if contributors and len(email_part) > 0:
            for c in contributors:
                distance = levenshtein_distance(c['login'].lower(), email_part, score_cutoff=10)
                if distance/len(email_part) < 0.33:
                    author_in_top_100 = True
                    break

        self._add_feature('author_in_top_100', author_in_top_100)
                
            
    def _mine_code_analysis(self, commit: Commit):
        self._add_feature('dmm_unit_complexity', commit.dmm_unit_complexity)
        self._add_feature('dmm_unit_interfacing', commit.dmm_unit_interfacing)
        self._add_feature('dmm_unit_size', commit.dmm_unit_size)


    def _mine_bag_of_words(self, commit: Commit):
        message = commit.msg.lower()
        message_escaped = self._escape_separators(message)
        title_escaped = self._escape_separators(message.split('\n')[0])

        for k in constants_and_configs.better_security_keywords:
            keyword_in_title = k in title_escaped
            keyword_in_message = k in message_escaped

            self._add_feature(f'{k}_in_title', keyword_in_title)
            self._add_feature(f'{k}_in_message', keyword_in_message)


    def _mine_history_based(self, history, index):
        commits_prev_7_days = []
        commits_next_7_days = []
        commits_next_30_days = []
        first_merge_after = None
        offset_of_prev_merge = -50
        offset_of_next_merge = 50

        for offset in range(-50, 50):
            i = index + offset
            if i < 0:
                continue
            if i >= len(history): 
                continue
            if not history[i]:
                continue

            delay_in_hours = (history[index].author_date - history[i].author_date).total_seconds() / 3600
            if 0 < delay_in_hours and delay_in_hours < 7*24:
                commits_next_7_days.append(history[i])
            if 0 < delay_in_hours and delay_in_hours < 30*24:
                commits_next_30_days.append(history[i])
            if -7*24 < delay_in_hours and delay_in_hours < 0:
                commits_prev_7_days.append(history[i])

            if offset > 0 and first_merge_after == None and history[i].merge:
                first_merge_after = history[i].author_date
            if offset > 0 and offset_of_next_merge > offset and history[i].merge:
                offset_of_next_merge = offset

            if offset < 0 and history[i].merge:
                offset_of_prev_merge = offset

        if index - 1 < 0:
            time_to_prev_commit = 0
        else: 
            time_to_prev_commit = (history[index].author_date - history[index - 1].author_date).total_seconds() / 3600

        if index + 1 >= len(history):
            time_to_next_commit = 0
        else: 
            time_to_next_commit = (history[index].author_date - history[index + 1].author_date).total_seconds() / 3600

        self._add_feature('commits_prev_7_days', len(commits_prev_7_days))
        self._add_feature('commits_next_7_days', len(commits_next_7_days))
        self._add_feature('commits_next_30_days', len(commits_next_30_days))

        if first_merge_after != None:
            delay_in_hours = (history[index].author_date - first_merge_after).total_seconds() / 3600
            self._add_feature('time_to_next_merge', delay_in_hours)
        else:
            self._add_feature('time_to_next_merge', float('nan'))

        self._add_feature('commits_to_next_merge', offset_of_next_merge)
        self._add_feature('commits_since_last_merge', -offset_of_prev_merge)

        self._add_feature('time_to_prev_commit', time_to_prev_commit)
        self._add_feature('time_to_next_commit', time_to_next_commit)

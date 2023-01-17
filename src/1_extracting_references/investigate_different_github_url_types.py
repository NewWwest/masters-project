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

from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader

if __name__ == '__main__':
    path_to_nvd_dump_json_files = 'path_to_nvd_dump_json_files'
    path_to_osv_data_dump = 'path_to_osv_data_dump'
    path_to_ghsa_data_dump = 'path_to_ghsa_data_dump'

    nvd = NvdLoader(path_to_nvd_dump_json_files)
    osv = OsvLoader(path_to_osv_data_dump)
    ghsa = OsvLoader(path_to_ghsa_data_dump)
    omni = OmniLoader(nvd, osv, ghsa)

    all = 0
    commits = 0
    commit = 0
    issues = 0
    issue = 0
    pull = 0
    compare = 0
    diff = 0
    blob = 0
    tree = 0
    releases = 0
    advisories = 0
    wiki = 0
    too_short = 0
    other = 0
    
    s_all = set()
    s_commits = set()
    s_commit = set()
    s_issues = set()
    s_issue = set()
    s_pull = set()
    s_compare = set()
    s_diff = set()
    s_blob = set()
    s_tree = set()
    s_releases = set()
    s_advisories = set()
    s_wiki = set()
    s_too_short = set()
    s_other = set()


    for report_id in omni.reports:
        refs = omni.references_from_report_list(omni.reports[report_id])
        github_ref = [x for x in refs if '/github.com/' in x]
        for url in github_ref:
            all+=1
            s_all.add(url)
            segments = url.split('/')

            if len(segments) < 7:
                too_short+=1
                s_too_short.add(url)
            elif '/commit/' in url:
                commit+=1
                s_commit.add(url)
            elif '/commits/' in url:
                commits+=1
                s_commits.add(url)
            elif '/issue/' in url:
                issue+=1
                s_issue.add(url)
            elif '/issues/' in url:
                issues+=1
                s_issues.add(url)
            elif '/pull/' in url:
                pull+=1
                s_pull.add(url)
            elif '/compare/' in url:
                compare+=1
                s_compare.add(url)
            elif '/diff/' in url:
                diff+=1
                s_diff.add(url)
            elif '/blob/' in url:
                blob+=1
                s_blob.add(url)
            elif '/tree/' in url:
                tree+=1
                s_tree.add(url)
            elif '/releases/' in url:
                releases+=1
                s_releases.add(url)
            elif '/security/advisories' in url or 'github.com/advisories' in url:
                advisories+=1
                s_advisories.add(url)
            elif 'wiki' in url:
                wiki+=1
                s_wiki.add(url)
            else:
                other+=1
                s_other.add(url)

    print('all', all,                   len(s_all))
    print('commits', commits,           len(s_commits))
    print('commit', commit,             len(s_commit))
    print('issues', issues,             len(s_issues))
    print('issue', issue,               len(s_issue))
    print('pull', pull,                 len(s_pull))
    print('compare', compare,           len(s_compare))
    print('diff', diff,                 len(s_diff))
    print('blob', blob,                 len(s_blob))
    print('tree', tree,                 len(s_tree))
    print('releases', releases,         len(s_releases))
    print('advisories', advisories,     len(s_advisories))
    print('wiki',  wiki,                len(s_wiki))
    print('too_short',  too_short,      len(s_too_short))
    print('other',  other,              len(s_other))

    # all 52953 30362
    # commits 560 311
    # commit 17152 8948
    # issues 10817 7288
    # issue 4 4
    # pull 3580 2053
    # compare 656 341
    # diff 1 1
    # blob 6683 4069
    # tree 2218 1866
    # releases 2678 1137
    # advisories 5735 2691
    # wiki 155 79
    # too_short 2617 1509
    # other 97 65
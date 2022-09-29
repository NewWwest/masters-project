from src.services.GithubProxy import GithubProxy
import json
import pandas as pd

input_path = '/Users/awestfalewicz/Private/data/advisory-database/advisories/'
checkpoint_path = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/results/checkpoints'
max_allowed_commits_in_link = 10
OUTPUT_PATH = 'output_path.json'

class FixMapper:
    def __init__(self):
        self.githubService = GithubProxy()
        self.mapping = {}



    def get_directly_linked_commits(self, report):
        links = [ref["url"] for ref in report.references if "/commit/" in ref["url"] and 'github.com' in ref["url"]]
        commit_infos = [self._commit_info_from_link(link) for link in links]
        return commit_infos


    def get_commits_from_linked_issues(self, report):
        mapping = []
        for ref in report.references:
            link = ref["url"]
            if 'github.com' in link and '/issues/' in link:
                result = self.githubService.commits_from_issue(link)
                if result:
                    mapping += result
        return mapping


    def get_commits_from_linked_pulls(self, report):
        mapping = []
        for ref in report.references:
            link = ref["url"]
            if 'github.com' in link and '/pull/' in link:
                result = self.githubService.commits_from_pull(link)
                if result:
                    mapping += result
        return mapping


    def get_commits_from_linked_compares(self, report):
        mapping = []
        for ref in report.references:
            link = ref["url"]
            if 'github.com' in link and ('/compare/' in link or '/diff/' in link):
                result = self.githubService.commits_from_diff(link)
                if result:
                    mapping += result
        return mapping


    def get_commits_with_vulnerability_ID_in_message(self, report, ids):
        segmented_urls = [ref['url'].split('#')[0].split('/') for ref in report.references if 'github.com' in ref['url']]
        repos = [f'{segments[3]}/{segments[4]}' for segments in segmented_urls if len(segments)>4]
        repos = list(set(repos))
        result = []
        for repo in repos:
            for id in ids:
                result += self.githubService.search_commits(repo, id)
        return result


    def _add_to_mapping_store(self, commits):
        for x in commits:
            if x in self.mapping:
                for commit_info in commits[x]:
                    duplicate = any([y for y in self.mapping[x] if y['sha']==commit_info['sha']])
                    if duplicate:
                        break
                    self.mapping[x].append(commit_info)
            else:
                self.mapping[x] = commits[x]


    def _add_commits_ref(self, commits, data_store, cve_id, ghsa_id, reason):
        if len(commits) > max_allowed_commits_in_link:
            print('WARN: too many commits to be considerred a fix')
            return

        for commit in commits:
            duplicate = any([x for x in data_store if x['sha']==commit['sha']])
            if duplicate:
                break
            commit['ghsa_id'] = ghsa_id
            commit['cve_id'] = cve_id
            commit['source'] = reason
            data_store.append(commit)


    def _commit_info_from_link(self, commit_link):
        segments = commit_link.split('/')
        result = {}
        result['owner'] = segments[3]
        result['repo'] = segments[4]
        result['sha'] = segments[6][:10]
        return result


    def import_commits_linked_in_vulnerabilities(self, reports_loader, limit=None, checkpoint = None):
        current_mapping = {}
        if checkpoint != None:
            with open(f'{checkpoint_path}/mapping_checkpoint_{checkpoint}.json', 'r') as f:
                current_mapping = json.load(f)

        index = -1
        for rep in reports_loader.alphabetical_order:
            index += 1
            if checkpoint != None and index <= checkpoint:
                continue

            print(index, rep)
            report = reports_loader.reports[rep]
            raw_report = reports_loader.reports_raw_data[rep] 
            if 'aliases' in raw_report and len(raw_report['aliases']) > 0:
                cve_id = raw_report['aliases'][0].lower()
            else:
                cve_id = ''
            ghsa_id = rep.lower()

            found_commits = []
            directly_linked = self.get_directly_linked_commits(report)
            self._add_commits_ref(directly_linked, found_commits, cve_id , ghsa_id, 'direct-link')

            linked_issues = self.get_commits_from_linked_issues(report)
            self._add_commits_ref(linked_issues, found_commits, cve_id , ghsa_id, 'issue-link')

            linked_pulls = self.get_commits_from_linked_pulls(report)
            self._add_commits_ref(linked_pulls, found_commits, cve_id , ghsa_id, 'pull-link')

            linked_diffs = self.get_commits_from_linked_compares(report)
            self._add_commits_ref(linked_diffs, found_commits, cve_id , ghsa_id, 'diff-link')

            id_in_the_message = self.get_commits_with_vulnerability_ID_in_message(report, [cve_id , ghsa_id])
            self._add_commits_ref(id_in_the_message, found_commits, cve_id , ghsa_id, 'scan-link')

            
            if len(found_commits) > 0:
                current_mapping[rep] = found_commits
            
            if limit != None and index >= limit:
                break

            if index % 500 == 0:
                with open(f'{checkpoint_path}/mapping_checkpoint_{index}.json', 'w') as f:
                    json.dump(current_mapping, f)

        self._add_to_mapping_store(current_mapping)
        return current_mapping


    def import_commits_linked_in_security_issues(self, path_to_issue_ref_file):
        current_mapping = {}
        data = pd.read_csv(path_to_issue_ref_file)
        for index, row in data.iterrows():
            issue_html_id = row["issue_url"].split('/')[-1]
            issue_key = f'ISSUE-{issue_html_id}'
            result = {}
            segments = row['repo_full_name'].split('/')
            if row['reference_type'] == 'ref_commit':
                reason = 'referenced-security-issue'
            elif row['reference_type'] == 'ref_ref_commit':
                reason = 'cross-referenced-security-issue'
            else:
                continue

            if 'http' not in row['reference_value']:
                sha = row['reference_value']
            else:
                temp = row['reference_value'].split('/') 
                sha = temp[-1]

            result['owner'] = segments[0]
            result['repo'] = segments[1]
            result['sha'] = sha
            result['ghsa_id'] = issue_key
            result['cve_id'] = issue_key
            result['source'] = reason
            if issue_key not in current_mapping:
                current_mapping[issue_key] = []

            current_mapping[issue_key].append(result)
        
        current_mapping_filterred = {}
        for x in current_mapping:
            if len(current_mapping[x]) <= max_allowed_commits_in_link:
                current_mapping_filterred[x] = current_mapping[x]

        self._add_to_mapping_store(current_mapping_filterred)
        return current_mapping_filterred

    def save(self, out_path):
        with open(out_path, 'w') as f:
            json.dump(self.mapping, f, indent=2)


if __name__ == '__main__':
    # path_to_issue_ref_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/src/security_issues_to_vulnerabilities/data/most_starred/filterred_refs.csv'
    mapper = FixMapper()
    # mapper.import_commits_linked_in_security_issues(path_to_issue_ref_file)
    # repo = OsvReportsLoader().load(input_path)
    # mapped_commits = mapper.import_commits_linked_in_vulnerabilities(repo, checkpoint=122000)
    mapper.save('sec-star.json')

from src.loaders.temp.OsvReportsLoader import OsvReportsLoader
from src.services.GithubProxy import GithubProxy
import json
import pandas as pd
import os

input_path = '/Users/awestfalewicz/Private/data/advisory-database/advisories'
checkpoint_path = 'results/checkpoint_for_issues'

class FixMapper:
    def __init__(self):
        self.githubService = GithubProxy()
        self.mapping = {}


    def get_entity_stuff(self, report):
        mapping = []
        for ref in report.references:
            link = ref["url"]
            if 'github.com' in link and ('/issues/' in link or '/pull/' in link):
                result = self.githubService.get_pull_or_issue(link)
                if result:
                    mapping.append(result)
        return mapping


    def download_issues_and_pulls(self, reports_loader, checkpoint = None):
        current_mapping = {}
        if checkpoint != None:
            with open(f'{checkpoint_path}/issue_checkpoint_{checkpoint}.json', 'r') as f:
                current_mapping = json.load(f)

        index = -1
        for rep in reports_loader.alphabetical_order:
            index += 1
            if checkpoint != None and index <= checkpoint:
                continue
            
            print(index, rep)
            report = reports_loader.reports[rep]

            try:
                issue_data = self.get_entity_stuff(report)
                if len(issue_data) != 0:
                    current_mapping[rep] = issue_data
            except Exception as e:
                print(e)

            if index % 250 == 0:
                with open(f'{checkpoint_path}/issue_checkpoint_{index}.json', 'w') as f:
                    json.dump(current_mapping, f)
                    current_mapping = {}

    def load_checkpoints(self, output):
        data = {}
        for root, subdirs, files in os.walk(checkpoint_path):
            for file in files:
                if file.endswith('.json') and file.startswith('issue_checkpoint_'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        temp = json.load(f)
                        data.update(temp)

        with open(output, 'w') as f:
            json.dump(data, f)


        


if __name__ == '__main__':
    mapper = FixMapper()
    # repo = OsvReportsLoader().load(input_path)
    # mapper.download_issues_and_pulls(repo, checkpoint = None)
    # mapper.load_checkpoints('all_issues_and_pulls_linked_to_vulnerabilities.json')

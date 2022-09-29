import json
from dateutil import parser

from src.loaders.temp.OsvReportsLoader import OsvReportsLoader
import src.rq4.constants_and_configs  as constants_and_configs

input_file = 'all_issues_and_pulls_linked_to_vulnerabilities.json'
security_label_keywords = ['secur', 'vulnerab', 'exploit']


ghsa_advisory_path = '/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewed/'
ghsa_advisory = OsvReportsLoader().load(ghsa_advisory_path)


with open(input_file, 'r') as f:
    data = json.load(f)


all = 0
positive = 0
per_topic = {}
for x in constants_and_configs.cwe_titles:
    per_topic[x]=set()

for ghsa in data:
    if ghsa not in ghsa_advisory.reports_raw_data:
        continue

    report = ghsa_advisory.reports_raw_data[ghsa]
    report_published = parser.parse(report['published'])
    all +=1
    for issue in data[ghsa]:
        all_text = ''
        if 'title' in issue and issue['title']:
            all_text += issue['title']
            all_text += ' '
        if 'body' in issue and issue['body']:
            all_text += issue['body']
            all_text += ' '

        if 'comments_data' in issue:
            for c in issue['comments_data']:
                created = parser.parse(c['created_at'])
                if created < report_published:
                    all_text += c['body']
                    all_text += ' '


        if 'timeline_data' in issue:
            for ti in issue['timeline_data']:
                event_date = None

                if ti['event'] == 'commented':
                    event_date = parser.parse(ti['created_at'])
                    text = ti['body']

                elif ti['event'] == 'committed':
                    event_date = parser.parse(ti['committer']['date'])
                    text = ti['message']

                elif ti['event'] == 'cross-referenced': 
                    event_date = parser.parse(ti['created_at'])
                    text = ti['source']['issue']['title']
                    
                elif ti['event'] == 'labeled': 
                    event_date = parser.parse(ti['created_at'])
                    text = ti['label']['name']

                elif ti['event'] == 'renamed':
                    event_date = parser.parse(ti['created_at'])
                    text = ti['rename']['from']
                    text = ti['rename']['to']

                elif ti['event'] == 'reviewed':
                    event_date = parser.parse(ti['submitted_at'])
                    text = ti['body']

                else:
                    ignored = ['mentioned',]
                    print(ti['event'])
                    # 'commit-commented'
                    # 'line-commented'

        
        for x in constants_and_configs.cwe_titles:
            if x in json.dumps(issue):
                positive += 1
                per_topic[x].add(ghsa)
                # print(x)
    if all%200==1:
        print(positive/all)

for x in constants_and_configs.cwe_titles:
    print(x, len(x))
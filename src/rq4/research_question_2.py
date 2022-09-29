import json
from unittest import result
from dateutil import parser

from src.loaders.temp.OsvReportsLoader import OsvReportsLoader

input_file = 'all_issues_and_pulls_linked_to_vulnerabilities.json'
security_label_keywords = ['secur', 'vulnerab', 'exploit']


ghsa_advisory_path = '/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewd/2018/'
ghsa_advisory = OsvReportsLoader().load(ghsa_advisory_path)


with open(input_file, 'r') as f:
    data = json.load(f)

def _check_single_issue(issue):
    if 'labels' in issue and len(issue['labels']) > 0:
        labels = issue['labels']
        for label in labels:
            for keyword in security_label_keywords:
                if keyword in label['name'].lower():
                    return True

def issue_has_a_sec_label(issue):
    result = _check_single_issue(issue)
    if result:
        return result
    
    cross_referenced = [x for x in issue['timeline_data'] if x['event'] == 'cross-referenced']
    cross_referenced_issues = [x['source']['issue'] for x in cross_referenced if x['source']['type'] == 'issue']
    cross_referenced_issues_in_the_same_repo = [x for x in cross_referenced_issues if x['repository']['full_name'] == repo_full_name]
    for cross_referenced in cross_referenced_issues_in_the_same_repo:
        result = _check_single_issue(cross_referenced)
        if result:
            return result

    return False




def created_before_vulnerability(issue, report):
    created_at = parser.parse(issue['created_at'])
    published = parser.parse(report['published'])
    if created_at < published:
        return True
        
    return False

def add_to_store(store, ecosystems, ghsa_id):
    store['all'].add(ghsa_id)
    for ecosystem in ecosystems:
        if ecosystem in store:
            store[ecosystem].add(ghsa_id)



count = {}
count['all'] = set()
count['npm'] = set()
count['pypi'] = set()
count['maven'] = set()

has_issue = {}
has_issue['all'] = set()
has_issue['npm'] = set()
has_issue['pypi'] = set()
has_issue['maven'] = set()

has_security_label = {}
has_security_label['all'] = set()
has_security_label['npm'] = set()
has_security_label['pypi'] = set()
has_security_label['maven'] = set()
created_at_before_vulnerability_publish = {}
created_at_before_vulnerability_publish['all'] = set()
created_at_before_vulnerability_publish['npm'] = set()
created_at_before_vulnerability_publish['pypi'] = set()
created_at_before_vulnerability_publish['maven'] = set()


for ghsa_id in ghsa_advisory.reports_raw_data:
    repo_full_name = 'swagger-api/swagger-ui'
    report = ghsa_advisory.reports_raw_data[ghsa_id]
    ecosystems = set([x['package']['ecosystem'].lower() for x in report['affected']])
    add_to_store(count, ecosystems, ghsa_id)
    if ghsa_id not in data:
        continue

    add_to_store(has_issue, ecosystems, ghsa_id)
    issues = data[ghsa_id]
    report = ghsa_advisory.reports_raw_data[ghsa_id]

    for issue in issues:
        issue_first = created_before_vulnerability(issue, report)
        add_to_store(created_at_before_vulnerability_publish, ecosystems, ghsa_id)
        if issue_first:
            issue_security = issue_has_a_sec_label(issue)
            if issue_security:
                add_to_store(has_security_label, ecosystems, ghsa_id)


total = len(ghsa_advisory.reports_raw_data)
print(total)
for x in count:
    print(x)
    print(len(has_security_label[x]), len(has_security_label[x])/len(count[x]))
    print(len(created_at_before_vulnerability_publish[x]), len(created_at_before_vulnerability_publish[x])/len(count[x]))
    print(len(has_issue[x]), len(has_issue[x])/len(count[x]))
    print(len(count[x]))

# # Results:


# # Entire Dataset:
# Total vulnerabilities: 178083
# ecosystem | before&security label | before | with issue link | with ecosystem
# all 922 8974 8974 178083
# npm 47 595 595 2498
# pypi 57 322 322 1137
# maven 31 323 323 133

# # GitHub-reviewed dataset: 
# Total vulnerabilities: 7444
# all
# 253 0.03398710370768404
# 2028 0.2724341751746373
# 2028 0.2724341751746373
# 7444
# npm
# 48 0.01922306768121746
# 595 0.2382859431317581
# 595 0.2382859431317581
# 2497
# pypi
# 57 0.05013192612137203
# 322 0.2832014072119613
# 322 0.2832014072119613
# 1137
# maven
# 31 0.02318623784592371
# 323 0.24158563949139866
# 323 0.24158563949139866
# 1337
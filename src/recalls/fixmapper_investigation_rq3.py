import sys
sys.path.insert(0, r'/Users/awestfalewicz/Projects/masters-project/')

import src.utils.utils as Utitls
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
import statistics
import json
import dateutil.parser as parser
import pandas as pd

input_data_location = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/results/checkpoints_fixMapper'
security_label_keywords  = ['secur', 'vulnerab', 'exploit']
ECOSYSTEM = None

def security_related(label_name):
    for keyword in security_label_keywords:
        if keyword in label_name:
            return True
    return False



def load_creation_dates(data_files, omni):
    
    all_references_delays = []
    issue_delays = []
    commit_delays = []

    seen_reports = []

    for data_file in data_files:
        with open(data_file, 'r') as f:
            data = json.load(f)

        for report_id in data:
            if ECOSYSTEM != None:
                ecosystems = omni.ecosystems_of_a_report(report_id)
                if not ecosystems:
                    continue
                
                if ECOSYSTEM not in ecosystems:
                    continue

            seen_reports.append(report_id)
            for ref in data[report_id]:
                if ref['reference_type'] == 'issue':
                    issue = json.loads(ref['data'])
                    creation_date = parser.parse(issue['created_at'])
                    publish_data = omni.publish_date_of_report(report_id)
                    if publish_data == None:
                        continue
                    publish_date = min([ x['publish_date'] for x in publish_data])
                    if publish_date.year < 2017:
                        continue
                    res = {
                        'issue_url': issue['url'],
                        'issue_create_date': creation_date,
                        'report_id': report_id,
                        'report_publish_date': publish_date,
                        'diff': (publish_date - creation_date).days
                    }
                    issue_delays.append(res)
                    all_references_delays.append(res)

    df = pd.DataFrame(parsed_dates)
    df = df.drop_duplicates(['issue_url', 'report_id'])
    return seen_reports, parsed_issues, df


def scan_for_security_issues(parsed_issues, issues_df):
    diff_to_security_labelling = []

    for i, x in issues_df.iterrows():
        issue = parsed_issues[x['issue_url']]
        if 'labels' in issue:
            label_names = [xx['name'].lower() for xx in issue['labels']]
            has_security_ralted_label = False
            for label_name in label_names:
                if security_related(label_name):
                    has_security_ralted_label = True
                    break
            if not has_security_ralted_label:
                continue    

            labelled_events = [evt for evt in issue['timeline_data'] if evt['event']=='labeled' and security_related(evt['label']['name'].lower())]
            actual_labelling_date = None
            if len(labelled_events) == 0:
                actual_labelling_date = parser.parse(issue['created_at'])
            else:
                labelling_dates = [parser.parse(evt['created_at']) for evt in labelled_events]
                actual_labelling_date = min(labelling_dates)

            res = {
                'issue_url': x['issue_url'],
                'issue_create_date': x['issue_create_date'],
                'issue_labelling_date': actual_labelling_date,
                'report_id': x['report_id'],
                'report_publish_date': x['report_publish_date'],
                'diff': (x['report_publish_date']-actual_labelling_date).days
            }
            diff_to_security_labelling.append(res)


    df = pd.DataFrame(diff_to_security_labelling)
    return df


def main():
    nvd = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    osv = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    ghsa = OsvLoader('/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)
    data_files = Utitls.get_files_in_from_directory(input_data_location, '.json')
    seen_reports, issue_objects, all_issues_df = load_creation_dates(data_files, omni)
    security_issues_df = scan_for_security_issues(issue_objects, all_issues_df)

    # Percentage of issues with security labels
    all_issues = set(all_issues_df['issue_url'])
    security_issues = set(security_issues_df['issue_url'])
    percentage = len(security_issues)/len(all_issues)
    print(f'{percentage}({len(security_issues)}/{len(all_issues)}) of issues have security related labels')


    all_issues_by_report_id = all_issues_df.groupby('report_id')
    all_issues_largest_diff =[]
    for report_id, iss1 in all_issues_by_report_id:
        all_issues_largest_diff.append(statistics.mean(iss1['diff']))
    sec_issues_by_report_id = security_issues_df.groupby('report_id')
    sec_issues_largest_diff =[]
    for report_id, iss2 in sec_issues_by_report_id:
        sec_issues_largest_diff.append(statistics.mean(iss2['diff']))

    mean_delay_for_all = statistics.mean(all_issues_largest_diff)
    medn_delay_for_all = statistics.median(all_issues_largest_diff)
    mean_delay_for_sec = statistics.mean(sec_issues_largest_diff)
    medn_delay_for_sec = statistics.median(sec_issues_largest_diff)
    print(f'The average/median delay of vulnerabilities compared to linked issues is {mean_delay_for_all}/{medn_delay_for_all}')
    print(f'The average/median delay of vulnerabilities compared to security issues is {mean_delay_for_sec}/{medn_delay_for_sec}')

    percentage_of_vulnerabilities_cought = len(sec_issues_by_report_id)/len(set(seen_reports))
    percentage_of_vulnerabilities_with_issues_cought = len(sec_issues_by_report_id)/len(all_issues_by_report_id)

    print('percentage_of_vulnerabilities_cought', percentage_of_vulnerabilities_cought)
    print('percentage_of_vulnerabilities_with_issues_cought', percentage_of_vulnerabilities_with_issues_cought)

    # all_issues_df.to_csv('issues_under_vulnerabilities_and_delays.csv', index=False)
    # security_issues_df.to_csv('issues_security_under_vulnerabilities_and_delays.csv', index=False)



if __name__ == '__main__':
    ECOSYSTEM='pypi'
    main()

# All
# 0.09706814580031696(735/7572) of issues have security related labels
# The average/median delay of vulnerabilities compared to linked issues is 141.15127570449351/28.0
# The average/median delay of vulnerabilities compared to security issues is 73.86770072992701/7.0
# DEBUG 224246 22807
# percentage_of_vulnerabilities_cought 0.04805542158109352
# percentage_of_vulnerabilities_with_issues_cought 0.10434120335110435

# npm
# 0.06906906906906907(23/333) of issues have security related labels
# The average/median delay of vulnerabilities compared to linked issues is 218.05571847507332/33.0
# The average/median delay of vulnerabilities compared to security issues is 258.7391304347826/102.0
# DEBUG 224246 1728
# percentage_of_vulnerabilities_cought 0.02662037037037037
# percentage_of_vulnerabilities_with_issues_cought 0.06744868035190615

# pypi
# 0.1238390092879257(40/323) of issues have security related labels
# The average/median delay of vulnerabilities compared to linked issues is 182.0841530054645/41
# The average/median delay of vulnerabilities compared to security issues is -1.735042735042735/0
# DEBUG 224246 3217
# percentage_of_vulnerabilities_cought 0.036369288156667706
# percentage_of_vulnerabilities_with_issues_cought 0.12786885245901639

# maven
# 0.09881422924901186(25/253) of issues have security related labels
# The average/median delay of vulnerabilities compared to linked issues is 125.93148148148148/38.5
# The average/median delay of vulnerabilities compared to security issues is 78.48/4.0
# DEBUG 224246 929
# percentage_of_vulnerabilities_cought 0.05382131324004306
# percentage_of_vulnerabilities_with_issues_cought 0.09259259259259259
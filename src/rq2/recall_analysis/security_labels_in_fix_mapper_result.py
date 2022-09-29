from ast import Index
import src.src2.utils.utils as Utitls
from src.src2.loaders.NvdLoader import NvdLoader
from src.src2.loaders.OsvLoader import OsvLoader
from src.src2.loaders.OmniLoader import OmniLoader
import json
import dateutil.parser as parser
import pandas as pd

input_data_location = 'results/checkpoints_fixMapper_new'
security_label_keywords  = ['secur', 'vulnerab', 'exploit']

def security_related(label_name):
    for keyword in security_label_keywords:
        if keyword in label_name:
            return True
    return False


def load_creation_dates(data_files, omni):
    parsed_dates = []
    parsed_issues = {}
    for data_file in data_files:
        print('Processing', data_file)
        with open(data_file, 'r') as f:
            data = json.load(f)

        for report_id in data:
            for ref in data[report_id]:
                if ref['reference_type'] == 'issue':
                    issue = json.loads(ref['data'])
                    parsed_issues[issue['url']] = issue
                    creation_date = parser.parse(issue['created_at'])
                    publish_data = omni.publish_date_of_report(report_id)
                    if publish_data == None:
                        continue
                    for x in publish_data:
                        res = {
                            'issue_url': issue['url'],
                            'issue_create_date': creation_date,
                            'report_id': x['report_id'],
                            'report_publish_date': x['publish_date'],
                            'diff': (x['publish_date'] - creation_date).days
                        }
                        parsed_dates.append(res)

    df = pd.DataFrame(parsed_dates)
    df = df.drop_duplicates(['issue_url', 'report_id'])
    return parsed_issues, df


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

    issue_objects, all_issues_df = load_creation_dates(data_files, omni)
    security_issues_df = scan_for_security_issues(issue_objects, all_issues_df)

    all_issues = set(all_issues_df['issue_url'])
    security_issues = set(security_issues_df['issue_url'])
    percentage = len(security_issues)/len(all_issues)

    mean_delay_for_all = all_issues_df['diff'].mean()
    medn_delay_for_all = all_issues_df['diff'].median()
    mean_delay_for_sec = security_issues_df['diff'].mean()
    medn_delay_for_sec = security_issues_df['diff'].median()

    all_issues_df.to_csv('issues_under_vulnerabilities_and_delays.csv', index=False)
    security_issues_df.to_csv('issues_security_under_vulnerabilities_and_delays.csv', index=False)

    print(f'The average/median delay of vulnerabilities compared to linked issues is {mean_delay_for_all}/{medn_delay_for_all}')
    print(f'The average/median delay of vulnerabilities compared to security issues is {mean_delay_for_sec}/{medn_delay_for_sec}')
    print(f'{percentage}({len(security_issues)}/{len(all_issues)}) of issues have security related labels')


if __name__ == '__main__':
    main()
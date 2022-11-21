import sys
sys.path.insert(0, r'D:\Projects\aaa')


from statistics import mean, median
from collections import Counter
import regex as re
from src.utils.utils import get_files_in_from_directory
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
import json
import dateutil.parser as parser
import pandas as pd
import matplotlib.pyplot as plt
import random

commits = set()
commits_data = {}
input_data_location = 'results/checkpoints_fixMapper'
keywords_regexes_path = r'src\rq3\bigquery\keywords_to_upload.csv'
all_labels = []

def safe_add(text, stuff):
    return text + ' ' + stuff if stuff else text


def get_date(ref):
    if ref['reference_type'] == 'commit':
        ref_data = json.loads(ref['data'])
        date1 = parser.parse(ref_data['commit']['committer']['date'])
        date2 = parser.parse(ref_data['commit']['author']['date'])
        first = date1 if date1 < date2 else date2
        return first

    if ref['reference_type'] == 'issue':
        ref_data = json.loads(ref['data'])
        date2 = parser.parse(ref_data['created_at'])
        return date2

    if ref['reference_type'] == 'compare':
        ref_data = json.loads(ref['data'])
        dates = [] 
        for x in ref_data['commits']:
            dates.append(parser.parse(x['commit']['author']['date']))
            dates.append(parser.parse(x['commit']['committer']['date']))
        first = min(dates)
        return first 


def security_related(ref, regexes):
    text = ''
    if ref['reference_type'] == 'commit':
        asd = json.loads(ref['data'])
        text = safe_add(text, asd['commit']['message'])
    elif ref['reference_type'] == 'issue':
        asd = json.loads(ref['data'])
        text = safe_add(text, asd['title'])
        text = safe_add(text, asd['body'])
        for t in asd['timeline_data']:
            if t['event']=='renamed':
                text = safe_add(text, t['rename']['from'])
                text = safe_add(text, t['rename']['to'])
            elif t['event']=='commented':
                text = safe_add(text, t['body'])
            elif t['event']=='committed':
                text = safe_add(text, t['message'])

        if 'pull_request_data' in asd:
            text = safe_add(text, asd['pull_request_data']['title'])
            text = safe_add(text, asd['pull_request_data']['body'])
        if 'pull_request_commits' in asd:
            for c in asd['pull_request_commits']:
                text = safe_add(text, c['commit']['message'])
        if 'pull_request_comments' in asd:
            for c in asd['pull_request_comments']:
                text = safe_add(text, c['body'])
    elif ref['reference_type'] == 'compare':
        asd = json.loads(ref['data'])
        for c in asd['commits']:
            text = safe_add(text, c['commit']['message'])

    if text:
        for req in regexes:
            if regexes[req].search(text):
                return True

    return False


def add_date(ref_aliases, date, by_cve):
    for alias_id in ref_aliases:
        if alias_id not in by_cve:
            by_cve[alias_id] = []
        by_cve[alias_id].append(date)


def load_creation_dates(ecosystem = None):
    nvd = NvdLoader(r'D:\Projects\VulnerabilityData\new_nvd')
    osv = OsvLoader(r'D:\Projects\VulnerabilityData\new_osv')
    ghsa = OsvLoader(r'D:\Projects\VulnerabilityData\advisory-database/advisories/github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)
    publish_dates_by_cve = {}
    for x in omni.reports:
        asd = omni.publish_date_of_report(x)
        asd = [x['publish_date'] for x in asd]
        first = min(asd)
        if first.year < 2017:
            continue

        if ecosystem == None:
            publish_dates_by_cve[x] = first
        
        if ecosystem != None:
            ecosystems = set([xx.lower() for xx in omni.ecosystems_of_a_report(x)])
            if ecosystem in ecosystems:
                publish_dates_by_cve[x] = first

    return publish_dates_by_cve


def dates_from_github():
    with open(keywords_regexes_path, 'r') as f:
        regexes = {x.strip(): re.compile(x.strip()) for x in f.readlines()}
    data_files = get_files_in_from_directory(input_data_location, '.json')
    security_issues = {}

    for data_file in data_files:
        # print(f'Processing... {data_file}')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for report_id in data:
            for ref in data[report_id]:
                ref_aliases = json.loads(ref['aliases'])
                
                if security_related(ref, regexes):
                    date = get_date(ref)
                    if date:
                        add_date(ref_aliases, date, security_issues)

    return security_issues
    



def main(ecosystem = None):
    seconds_in_a_day = 24*3600
    publish_dates_by_cve = load_creation_dates(ecosystem)
    security_issues = dates_from_github()
    
    keys = set(publish_dates_by_cve.keys()).intersection(set(security_issues.keys()))
    issue_delays = []
    for report_id in keys:
        delays2 = [publish_dates_by_cve[report_id]-ref_date for ref_date in security_issues[report_id]]
        delays2 = [x.total_seconds()/seconds_in_a_day for x in delays2]
        issue_delays += delays2
    
    print('===issue_delays===')
    print(len(issue_delays))
    print(mean(issue_delays))
    print(median(issue_delays))

    # xd = [-x for x in issue_delays]
    # plt.ylabel('Number of references')
    # plt.hist(xd,  bins=178, range=(-5*356, 356))
    # plt.xlabel('Days to/after disclosure')
    # plt.axis(ymin=0, xmin=-5*356, xmax=356)
    # plt.show()

    # xd = [-x for x in issue_delays]
    # plt.hist(xd,  bins=110, range=(-100, 10))
    # plt.ylabel('Number of references')
    # plt.xlabel('Days to/after disclosure')
    # plt.axis(ymin=0, xmin=-100, xmax=10)
    # plt.show()


    all_reports = set()
    reports_with_refs = set()
    d0_reports = set()
    d7_reports = set()
    for x in publish_dates_by_cve:
        all_reports.add(x)
        if x not in keys:
            continue
        reports_with_refs.add(x)

        delays = [publish_dates_by_cve[x]-ref_date for ref_date in security_issues[x]]
        delays = [x.total_seconds()/seconds_in_a_day for x in delays]
        max_delay = max(delays)
        if max_delay > 0:
             d0_reports.add(x)
        if max_delay > 7:
             d7_reports.add(x)

    print(ecosystem)
    print(len(all_reports), len(reports_with_refs), len(d0_reports), len(d7_reports))
    print(len(d0_reports)/len(all_reports), len(d7_reports)/len(all_reports))
    print(len(d0_reports)/len(reports_with_refs), len(d7_reports)/len(reports_with_refs))


if __name__ == '__main__':
    main('npm')
    main('pypi')
    main('maven')
    main()
    print('ok')


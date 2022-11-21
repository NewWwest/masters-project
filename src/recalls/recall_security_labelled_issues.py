from statistics import mean, median
from collections import Counter
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
security_label_keywords  = ['secur', 'vulnerab', 'exploit']

def security_related(label_name):
    for keyword in security_label_keywords:
        if keyword in label_name:
            return True
    return False


def add_date(ref_aliases, date, by_cve):
    for alias_id in ref_aliases:
        if alias_id not in by_cve:
            by_cve[alias_id] = []
        by_cve[alias_id].append(date)


def load_creation_dates(ecosystem = None):
    nvd = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    osv = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    ghsa = OsvLoader('/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewed')
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
    all_issue_links = set()
    sec_issue_links = set()
    reports_with_sec_issues = set()
    data_files = get_files_in_from_directory(input_data_location, '.json')
    security_issues = {}

    for data_file in data_files:
        # print(f'Processing... {data_file}')
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for report_id in data:
            for ref in data[report_id]:
                ref_aliases = json.loads(ref['aliases'])
                if ref['reference_type'] == 'issue':
                    reports_with_sec_issues.add(report_id)
                    ref_data = json.loads(ref['data'])
                    all_issue_links.add(ref_data['url'])
                    date = security_label_date(ref_data)
                    if date:
                        add_date([report_id], date, security_issues)
                        sec_issue_links.add(ref_data['url'])

    return security_issues, sec_issue_links, all_issue_links, reports_with_sec_issues
    

def security_label_date(issue):
    if 'labels' not in issue:
        return None

    label_names = [xx['name'].lower() for xx in issue['labels']]
    has_security_ralted_label = False
    for label_name in label_names:
        if security_related(label_name):
            has_security_ralted_label = True
            break

    
    labelled_events = [evt for evt in issue['timeline_data'] if evt['event']=='labeled' and security_related(evt['label']['name'].lower())]

    actual_labelling_date = None
    if len(labelled_events) == 0 and not has_security_ralted_label:
        return None
    elif len(labelled_events) == 0 and has_security_ralted_label:
        actual_labelling_date = parser.parse(issue['created_at'])
    else:
        labelling_dates = [parser.parse(evt['created_at']) for evt in labelled_events]
        actual_labelling_date = min(labelling_dates)


    return actual_labelling_date




def main(ecosystem):
    seconds_in_a_day = 24*3600
    publish_dates_by_cve = load_creation_dates(ecosystem)
    security_issues, sec_issue_links, all_issue_links, reports_with_sec_issues = dates_from_github()
    # counts = Counter(all_labels)
    # counts_sorted = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
    # for k,v in counts_sorted.items():
    #     print(k, v)
    
    keys = set(publish_dates_by_cve.keys()).intersection(set(security_issues.keys()))
    reports_with_sec_issues_keys = set(reports_with_sec_issues).intersection(set(publish_dates_by_cve.keys()))
    issue_delays = []
    for report_id in keys:
        delays2 = [publish_dates_by_cve[report_id]-ref_date for ref_date in security_issues[report_id]]
        delays2 = [x.total_seconds()/seconds_in_a_day for x in delays2]
        issue_delays.append(max(delays2))
    
    print('===Issues with labels===')
    print(len(sec_issue_links), len(all_issue_links), len(sec_issue_links)/len(all_issue_links))

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
    ref_reports= set()
    d0_reports = set()
    d7_reports = set()
    for x in publish_dates_by_cve:
        all_reports.add(x)
        if x not in keys:
            continue
        
        ref_reports.add(x)
        delays = [publish_dates_by_cve[x]-ref_date for ref_date in security_issues[x]]
        delays = [x.total_seconds()/seconds_in_a_day for x in delays]
        max_delay = max(delays)
        if max_delay > 0:
             d0_reports.add(x)
        if max_delay > 7:
             d7_reports.add(x)

    print(ecosystem)
    print('===Recalls===')
    print(len(all_reports), len(ref_reports),len(reports_with_sec_issues_keys))
    print(len(d0_reports), len(d7_reports))
    print('VS refs')
    print(len(d0_reports)/len(ref_reports), len(d7_reports)/len(ref_reports))
    print('VS all reports')
    print(len(d0_reports)/len(all_reports), len(d7_reports)/len(all_reports))
    print('VS reports_with_labels')
    print(len(d0_reports)/len(reports_with_sec_issues_keys), len(d7_reports)/len(reports_with_sec_issues_keys))

if __name__ == '__main__':
    print()
    print()
    main(None)
    print()
    print()
    main('npm')
    print()
    print()
    main('pypi')
    print()
    print()
    main('maven')
    print('ok')

# ===issue_delays=== BASE
# 111
# 99.76245391224558
# 4.655
# 3580 56 46 28
# 0.8214285714285714 0.5
# 0.012849162011173185 0.00782122905027933
# ok

# ===issue_delays=== NO_ALIASES
# 52
# 75.98405982905983
# 4.550856481481482
# 3580 52 42 24
# 0.8076923076923077 0.46153846153846156
# 0.011731843575418994 0.0067039106145251395
# ok

# ===issue_delays=== MAX
# 52
# 75.98405982905983
# 4.550856481481482
# 3580 52 42 24
# 0.8076923076923077 0.46153846153846156
# 0.011731843575418994 0.0067039106145251395
# ok
# RESULTS:



# ===Issues with labels===
# 766 7572 0.101162176439514
# ===issue_delays===
# 917
# 90.07345595288582
# 8.883888888888889
# None
# ===Recalls===
# 117853 917 8782
# 769 484
# VS refs
# 0.8386041439476554 0.5278080697928026
# VS all reports
# 0.006525077851221437 0.004106811027296717
# VS reports_with_labels
# 0.08756547483488955 0.05511273058528809


# ===Issues with labels===
# 766 7572 0.101162176439514
# ===issue_delays===
# 42
# 272.9367757936508
# 102.32747685185186
# npm
# ===Recalls===
# 4294 42 650
# 38 32
# VS refs
# 0.9047619047619048 0.7619047619047619
# VS all reports
# 0.008849557522123894 0.007452258965999069
# VS reports_with_labels
# 0.05846153846153846 0.04923076923076923


# ===Issues with labels===
# 766 7572 0.101162176439514
# ===issue_delays===
# 75
# 14.46508950617284
# 11.846608796296296
# pypi
# ===Recalls===
# 4158 75 758
# 61 38
# VS refs
# 0.8133333333333334 0.5066666666666667
# VS all reports
# 0.01467051467051467 0.00913900913900914
# VS reports_with_labels
# 0.08047493403693931 0.05013192612137203


# ===Issues with labels===
# 766 7572 0.101162176439514
# ===issue_delays===
# 48
# 43.740640432098765
# 4.3373842592592595
# maven
# ===Recalls===
# 3307 48 488
# 38 20
# VS refs
# 0.7916666666666666 0.4166666666666667
# VS all reports
# 0.01149077713940127 0.0060477774417901425
# VS reports_with_labels
# 0.0778688524590164 0.040983606557377046
# ok
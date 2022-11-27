#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

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
import regex as re

input_data_location = r'data\2_extracted_references\data'
keywords_regexes_path = r'src\2_phrase_search\bigquery\keywords_v1.csv'

commits_data = {}
security_label_keywords  = ['secur', 'vulnerab', 'exploit']

# General
def _get_ref_date(ref):
    ref_data = ref['data_obj']
    if ref['reference_type'] == 'commit':
        date1 = parser.parse(ref_data['commit']['committer']['date'])
        date2 = parser.parse(ref_data['commit']['author']['date'])
        first = date1 if date1 < date2 else date2
        return first

    if ref['reference_type'] == 'issue':
        date2 = parser.parse(ref_data['created_at'])
        return date2

    if ref['reference_type'] == 'compare':
        dates = [] 
        for x in ref_data['commits']:
            dates.append(parser.parse(x['commit']['author']['date']))
            dates.append(parser.parse(x['commit']['committer']['date']))
        first = min(dates)
        return first 

# First commit helper
def _get_commits_from_issue(repo_owner, repo_name, issue_data):
    global commits_data

    repo_full_name = f'{repo_owner}/{repo_name}'
    issue_commits = []

    timeline_commits = _extract_commits_from_timeline(repo_owner, repo_name, issue_data['timeline_data'])
    issue_commits += [f'{repo_owner}/{repo_name}/{c}' for c in timeline_commits]
    if 'pull_request' in issue_data:
        if 'pull_request_commits' in issue_data:
            pr_commits = issue_data['pull_request_commits']
            for c in pr_commits:
                c_id = f'{repo_owner}/{repo_name}/{c["sha"]}'
                issue_commits.append(c_id)
    else:
        cross_referenced_issues = [x['source']['issue'] for x in issue_data['timeline_data'] if x['event']=='cross-referenced']
        cross_referenced_events_in_the_same_repo = [x for x in cross_referenced_issues if x['repository']['full_name']==repo_full_name]
        for referenced_issue in cross_referenced_events_in_the_same_repo:
            try:
                timeline_commits =  _extract_commits_from_timeline(repo_owner, repo_name, referenced_issue['timeline_data'])
                issue_commits += [f'{repo_owner}/{repo_name}/{c}' for c in timeline_commits]
                if 'pull_request' in referenced_issue:
                    if 'pull_request_commits' in referenced_issue:
                        pr_commits = referenced_issue['pull_request_commits']
                        for c in pr_commits:
                            c_id = f'{repo_owner}/{repo_name}/{c["sha"]}'
                            issue_commits.append(c_id)
            except:
                print('Failed to process ref issue')


    commits = []
    for x in list(set(issue_commits)):
        if x in commits_data:
            commits.append(commits_data[x])
    return commits


# First commit helper
def _extract_commits_from_timeline(repo_owner, repo_name, timeline):
    try:
        result = []
        referenced_commits = [x for x in timeline if 'commit_id' in x and x['commit_id'] != None]
        for c in referenced_commits:
            if 'html_url' in c:
                seg = c['html_url'].split('/')
                ref_repo_owner = seg[3]
                ref_repo_name = seg[4]
            else:
                seg = c['commit_url'].split('/')
                ref_repo_owner = seg[4]
                ref_repo_name = seg[5]

            if repo_owner == ref_repo_owner and repo_name == ref_repo_name:
                result.append(c['commit_id'])
        return result
    except:
        return []



# Security phrases helper
def _safe_add(text, stuff):
    return text + ' ' + stuff if stuff else text

# Security phrases helper
def _contains_secuirty_related_phrases(ref, regexes):
    text = ''
    asd = ref['data_obj']
    if ref['reference_type'] == 'commit':
        text = _safe_add(text, asd['commit']['message'])
    elif ref['reference_type'] == 'issue':
        text = _safe_add(text, asd['title'])
        text = _safe_add(text, asd['body'])
        for t in asd['timeline_data']:
            if t['event']=='renamed':
                text = _safe_add(text, t['rename']['from'])
                text = _safe_add(text, t['rename']['to'])
            elif t['event']=='commented':
                text = _safe_add(text, t['body'])
            elif t['event']=='committed':
                text = _safe_add(text, t['message'])

        if 'pull_request_data' in asd:
            text = _safe_add(text, asd['pull_request_data']['title'])
            text = _safe_add(text, asd['pull_request_data']['body'])
        if 'pull_request_commits' in asd:
            for c in asd['pull_request_commits']:
                text = _safe_add(text, c['commit']['message'])
        if 'pull_request_comments' in asd:
            for c in asd['pull_request_comments']:
                text = _safe_add(text, c['body'])
    elif ref['reference_type'] == 'compare':
        for c in asd['commits']:
            text = _safe_add(text, c['commit']['message'])

    if text:
        for req in regexes:
            if regexes[req].search(text):
                return True

    return False

# Security labels helper
def _security_related_label(label_name):
    for keyword in security_label_keywords:
        if keyword in label_name:
            return True
    return False


# Security labels helper
def _get_labelling_date(issue):
    if 'labels' not in issue:
        return None

    label_names = [xx['name'].lower() for xx in issue['labels']]
    has_security_ralted_label = False
    for label_name in label_names:
        if _security_related_label(label_name):
            has_security_ralted_label = True
            break

    if not has_security_ralted_label:
        return None    

    labelled_events = [evt for evt in issue['timeline_data'] if evt['event']=='labeled' and _security_related_label(evt['label']['name'].lower())]
    actual_labelling_date = None
    if len(labelled_events) == 0:
        actual_labelling_date = parser.parse(issue['created_at'])
    else:
        labelling_dates = [parser.parse(evt['created_at']) for evt in labelled_events]
        actual_labelling_date = min(labelling_dates)

    return actual_labelling_date
    
         




def load_creation_date(data, omni):
    for x in data.keys():
        dates = omni.publish_date_of_report(x)
        dates = [x['publish_date'] for x in dates]
        first = min(dates)
        if first.year < 2017:
            continue

        data[x]['creation_date'] = first

        
def load_ecosystem(data, omni):
    for x in data.keys():
        asd = omni.publish_date_of_report(x)
        asd = [x['publish_date'] for x in asd]
        first = min(asd)
        if first.year < 2017:
            continue

        ecosystems = list(set([xx.lower() for xx in omni.ecosystems_of_a_report(x)]))
        data[x]['ecosystem'] = ecosystems


def load_repositories(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['repositories'] = []
            continue

        repos = set()
        for ref in references_data[x]:  
            repos.add(f'{ref["repo_owner"]}/{ref["repo_name"]}')

        data[x]['repositories'] = list(repos)


def load_first_ref(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['first_ref'] = None
            continue

        dates = {}
        for ref in references_data[x]:  
            ref_date = _get_ref_date(ref)
            if ref_date != None:
                repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                if repo_full_name not in dates:
                    dates[repo_full_name]= []
                dates[repo_full_name].append(ref_date)

        if len(dates) == 0:
            data[x]['first_ref'] = None
        else:
            min_dates = [(k,min(v)) for k,v in dates.items()]
            data[x]['first_ref'] = min_dates


def load_first_issue(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['first_issue'] = None
            continue

        dates = {}
        for ref in references_data[x]:  
            if ref['reference_type'] != 'issue':
                continue

            ref_date = _get_ref_date(ref)
            if ref_date != None:
                repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                if repo_full_name not in dates:
                    dates[repo_full_name]= []
                dates[repo_full_name].append(ref_date)

        if len(dates) == 0:
            data[x]['first_issue'] = None
        else:
            min_dates = [(k,min(v)) for k,v in dates.items()]
            data[x]['first_issue'] = min_dates


def load_first_commit(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['first_commit'] = None
            continue

        dates = {}
        for ref in references_data[x]:  
            if ref['reference_type'] == 'issue':
                commits = _get_commits_from_issue(ref['repo_owner'], ref['repo_name'], ref['data_obj'])
                if commits == None or len(commits) ==0:
                    continue

                min_date = parser.parse(commits[0]['commit']['committer']['date'])
                for c in commits:
                    commiter_date = parser.parse(c['commit']['committer']['date'])
                    author_date = parser.parse(c['commit']['author']['date'])
                    if commiter_date < min_date:
                        min_date = commiter_date
                    if author_date < min_date:
                        min_date = author_date

                repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                if repo_full_name not in dates:
                    dates[repo_full_name]= []
                dates[repo_full_name].append(min_date)

            if ref['reference_type'] == 'commit' or ref['reference_type'] == 'compare':
                ref_date = _get_ref_date(ref)
                if ref_date != None:
                    repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                    if repo_full_name not in dates:
                        dates[repo_full_name]= []
                    dates[repo_full_name].append(ref_date)

        if len(dates) == 0:
            data[x]['first_commit'] = None
        else:
            min_dates = [(k,min(v)) for k,v in dates.items()]
            data[x]['first_commit'] = min_dates

           
def load_first_sec_phrase(data, references_data):
    with open(keywords_regexes_path, 'r') as f:
        regexes = {x.strip(): re.compile(x.strip()) for x in f.readlines()}

    for x in data.keys():
        if x not in references_data:
            data[x]['first_sec_phrase'] = None
            continue

        dates = {}
        for ref in references_data[x]: 
            if not _contains_secuirty_related_phrases(ref, regexes):
                continue

            ref_date = _get_ref_date(ref)
            if ref_date != None:
                repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                if repo_full_name not in dates:
                    dates[repo_full_name]= []
                dates[repo_full_name].append(ref_date)

        if len(dates) == 0:
            data[x]['first_sec_phrase'] = None
        else:
            min_dates = [(k,min(v)) for k,v in dates.items()]
            data[x]['first_sec_phrase'] = min_dates


def load_first_sec_label(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['first_sec_label'] = None
            continue

        dates = {}
        for ref in references_data[x]: 
            if ref['reference_type'] == 'issue':
                ref_date  = _get_labelling_date(ref['data_obj'])
                if ref_date != None:
                    repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
                    if repo_full_name not in dates:
                        dates[repo_full_name]= []
                    dates[repo_full_name].append(ref_date)

        if len(dates) == 0:
            data[x]['first_sec_label'] = None
        else:
            min_dates = [(k,min(v)) for k,v in dates.items()]
            data[x]['first_sec_label'] = min_dates




def main():
    global commits_data
    # Import data from vulnerability databases
    nvd = NvdLoader(r'data\new_nvd')
    osv = OsvLoader(r'data\new_osv')
    ghsa = OsvLoader(r'data\advisory-database\advisories\github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)
    
    # Import data from fix mapper result
    references_data = {}
    data_files = get_files_in_from_directory(input_data_location, '.json')
    for data_file in data_files:
        with open(data_file, 'r') as f:
            data_temp = json.load(f)

        for report_id in data_temp:
            for x in data_temp[report_id]:
                x['data_obj'] = json.loads(x['data'])
                x['data'] = None
        for report_id in data_temp:
            references_data[report_id] = data_temp[report_id]

    # Import data on additional commits
    with open(r'src\recalls\recall_analysis\commit_data_for_rq1.json', 'r') as f:
        commits_data = json.load(f)


    data = {x:{} for x in omni.all_ids}

    # Import creation dates
    load_creation_date(data, omni)

    # Load ecosystems
    load_ecosystem(data, omni)

    # Load repositories
    load_repositories(data, references_data)


    # Load first ref
    load_first_ref(data, references_data)

    # Load first issue
    load_first_issue(data, references_data)

    # Load first commit
    load_first_commit(data, references_data)

    # Load first ref with security phrases
    load_first_sec_phrase(data, references_data)

    # Load first issue with security label
    load_first_sec_label(data, references_data)

    # Load first commit classified as security related (Feature-based)
    # load_first_commit(data, omni)

    # Load first commit classified as security related (own-models)
    # load_first_issue(data, omni)

    # Load first commit classified as security related (VulFixMiner)
    # load_first_commit(data, omni)

    with open('revall_analysis.json', 'w') as f:
        json.dump(data, f, default=str)

if __name__ == '__main__':
    main()
    print('ok')



# def dates_from_github():
#     all_issue_links = set()
#     sec_issue_links = set()
#     reports_with_sec_issues = set()
#     data_files = get_files_in_from_directory(input_data_location, '.json')
#     security_issues = {}

#     for data_file in data_files:
#         # print(f'Processing... {data_file}')
#         with open(data_file, 'r') as f:
#             data = json.load(f)
        
#         for report_id in data:
#             for ref in data[report_id]:
#                 ref_aliases = json.loads(ref['aliases'])
#                 if ref['reference_type'] == 'issue':
#                     reports_with_sec_issues.add(report_id)
#                     ref_data = json.loads(ref['data'])
#                     all_issue_links.add(ref_data['url'])
#                     date = security_label_date(ref_data)
#                     if date:
#                         add_date([report_id], date, security_issues)
#                         sec_issue_links.add(ref_data['url'])

#     return security_issues, sec_issue_links, all_issue_links, reports_with_sec_issues
    


# def security_related(label_name):
#     for keyword in security_label_keywords:
#         if keyword in label_name:
#             return True
#     return False


# def add_date(ref_aliases, date, by_cve):
#     for alias_id in ref_aliases:
#         if alias_id not in by_cve:
#             by_cve[alias_id] = []
#         by_cve[alias_id].append(date)


# def security_label_date(issue):
#     if 'labels' not in issue:
#         return None

#     label_names = [xx['name'].lower() for xx in issue['labels']]
#     has_security_ralted_label = False
#     for label_name in label_names:
#         if security_related(label_name):
#             has_security_ralted_label = True
#             break

    
#     labelled_events = [evt for evt in issue['timeline_data'] if evt['event']=='labeled' and security_related(evt['label']['name'].lower())]

#     actual_labelling_date = None
#     if len(labelled_events) == 0 and not has_security_ralted_label:
#         return None
#     elif len(labelled_events) == 0 and has_security_ralted_label:
#         actual_labelling_date = parser.parse(issue['created_at'])
#     else:
#         labelling_dates = [parser.parse(evt['created_at']) for evt in labelled_events]
#         actual_labelling_date = min(labelling_dates)


#     return actual_labelling_date

#!/usr/bin/env python3
import sys
sys.path.insert(0, r'D:\Projects\aaadoc')

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

input_data_location = r'D:\Projects\aaa\results\checkpoints_fixMapper'
keywords_regexes_path = r'D:\Projects\aaadoc\src\2_phrase_search\bigquery\keywords_v2.csv'
vfm_classified_commits = r'D:\Projects\aaadoc\src\recalls\data\sap_result_training_excluded.json'
security_relevant_commits_file = r'src\recalls\data\security_relevant_commits.csv'
security_relevant_commits_info_file = r'src\recalls\data\security_relevant_commits_info.json'

fb_predictions_all = r'D:\Projects\aaadoc\src\recalls\data\prediction_all.json'
fb_predictions_npm = r'D:\Projects\aaadoc\src\recalls\data\prediction_npm.json'
fb_predictions_pypi = r'D:\Projects\aaadoc\src\recalls\data\prediction_pypi.json'
fb_predictions_mvn = r'D:\Projects\aaadoc\src\recalls\data\prediction_mvn.json'

commits_info_data = {}
commits_data = {}
security_label_keywords  = ['secur', 'vulnerab', 'exploit']

phrases_to_ids = {}

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

# General
def _get_ref_id(ref):
    return f'{ref["repo_owner"]}/{ref["repo_name"]}/{ref["reference_type"]}/{ref["reference_value"]}'

# General
def _commit_size(ref):
    global commits_info_data
    if ref["reference_type"] == 'commit':
        commit_id = f'{ref["repo_owner"]}/{ref["repo_name"]}/{ref["reference_value"]}'
        if commit_id in commits_info_data:
            return commits_info_data[commit_id]['commit_info']['stats']['total']
        else:
            return None
    else:
        return None

def _commit_size_by_commit_id(commit_id):
    global commits_info_data
    if commit_id in commits_info_data:
        return commits_info_data[commit_id]['commit_info']['stats']['total']
    else:
        return None

def _get_matched_ref_object(ref, ref_date):
    ref_id = _get_ref_id(ref)
    ref_commit_size = _commit_size(ref)
    repo_full_name = f'{ref["repo_owner"]}/{ref["repo_name"]}'
    matched_ref_obj = {
        'ref_id':ref_id,
        'ref_repo_full_name':repo_full_name,
        'ref_commit_size':ref_commit_size,
        'ref_date': ref_date
    }
    return matched_ref_obj
    
def _get_matched_ref_object_by_commit_id(commit_id, first):
    segments = commit_id.split('/')
    ref_id = f'{segments[0]}/{segments[1]}/commit/{segments[2]}'
    ref_commit_size = _commit_size_by_commit_id(commit_id)
    repo_full_name = f'{segments[0]}/{segments[1]}'
    matched_ref_obj = {
                    'ref_id':ref_id,
                    'ref_repo_full_name':repo_full_name,
                    'ref_commit_size':ref_commit_size,
                    'ref_date': first
                }
    
    return matched_ref_obj

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
def _contains_secuirty_related_phrases(ref, regexes, report_id):
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

    global phrases_to_ids
    found_phrases = False
    if text:
        for req in regexes:
            if regexes[req].search(text):
                if req not in phrases_to_ids:
                    phrases_to_ids[req] = set()
                phrases_to_ids[req].add(report_id)
                found_phrases = True

    return found_phrases

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


def load_severity(data, omni):
    for x in data.keys():
        asd = omni.publish_date_of_report(x)
        asd = [x['publish_date'] for x in asd]
        first = min(asd)
        if first.year < 2017:
            continue

        severity = omni.highest_severity(x)
        data[x]['severity'] = severity


def load_repositories(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['repositories'] = []
            continue

        repos = set()
        for ref in references_data[x]:  
            repos.add(f'{ref["repo_owner"]}/{ref["repo_name"]}')

        data[x]['repositories'] = list(repos)


def load_all_refs(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['all_refs'] = None
            continue

        dates = []
        for ref in references_data[x]:  
            ref_date = _get_ref_date(ref)
            if ref_date != None:
                matched_ref_obj = _get_matched_ref_object(ref, ref_date)
                dates.append(matched_ref_obj)

        if len(dates) == 0:
            data[x]['all_refs'] = None
        else:
            data[x]['all_refs'] = dates


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

        dates = []
        for ref in references_data[x]: 
            if not _contains_secuirty_related_phrases(ref, regexes, x):
                continue

            ref_date = _get_ref_date(ref)
            if ref_date != None:
                matched_ref_obj = _get_matched_ref_object(ref, ref_date)
                dates.append(matched_ref_obj)

        if len(dates) == 0:
            data[x]['first_sec_phrase'] = None
        else:
            data[x]['first_sec_phrase'] = dates


def load_first_sec_label(data, references_data):
    for x in data.keys():
        if x not in references_data:
            data[x]['first_sec_label'] = None
            continue

        dates = []
        for ref in references_data[x]: 
            if ref['reference_type'] == 'issue':
                ref_date  = _get_labelling_date(ref['data_obj'])
                if ref_date != None:
                    matched_ref_obj = _get_matched_ref_object(ref, ref_date)
                    dates.append(matched_ref_obj)

        if len(dates) == 0:
            data[x]['first_sec_label'] = None
        else:
            data[x]['first_sec_label'] = dates


def load_fb_commit_classification(data):
    global commits_info_data

    predictions = {}
    prediction_files = [
        fb_predictions_all, 
        fb_predictions_npm,
        fb_predictions_pypi,
        fb_predictions_mvn, 
    ]
    for prediction_file in prediction_files:
        with open(prediction_file, 'r') as f:
            temp = json.load(f)
            for x in temp:
                if x['actual_value']:
                    if x['commit_id'] not in predictions:
                        predictions[x['commit_id']] = x['prediction'] == 'True'
                    else:
                        if not predictions[x['commit_id']]:
                            predictions[x['commit_id']] = x['prediction'] == 'True'

    sighted_vulnerabilities = set()
    per_vulnerability = {}
    for commit_id in predictions:
        if commit_id not in commits_info_data:
            continue

        commit = commits_info_data[commit_id]['commit_info']
        sighted_vulnerabilities.update(commits_info_data[commit_id]['report_ids'])

        if predictions[commit_id]:
            date1 = parser.parse(commit['commit']['committer']['date'])
            date2 = parser.parse(commit['commit']['author']['date'])
            first = date1 if date1 < date2 else date2
        else:
            first = 'not_spotted'

        
        for report_id in commits_info_data[commit_id]['report_ids']:
            if report_id not in per_vulnerability:
                per_vulnerability[report_id]=[]

            matched_ref_obj = _get_matched_ref_object_by_commit_id(commit_id, first)
            per_vulnerability[report_id].append(matched_ref_obj)


    for x in sighted_vulnerabilities:
        data[x]['fb_commit_classification'] = per_vulnerability[x]



def load_vfm_commit_classification(data):
    global commits_info_data

    with open(vfm_classified_commits, 'r') as f:
        vfm_classified = json.load(f)

    sighted_vulnerabilities = set()
    per_vulnerability_msg = {}
    per_vulnerability_patch = {}
    per_vulnerability_combined = {}
    for x in vfm_classified:
        if not x['is_security_related']:
            continue

        if x['id'] not in commits_info_data:
            continue

        commit_id = x['id']
        commit = commits_info_data[x['id']]['commit_info']
        sighted_vulnerabilities.update(commits_info_data[x['id']]['report_ids'])

        date1 = parser.parse(commit['commit']['committer']['date'])
        date2 = parser.parse(commit['commit']['author']['date'])
        first = date1 if date1 < date2 else date2
        if x['msg_prob'][0] >= 0.5 :
            spotted_date = first
        else:
            spotted_date = 'not_spotted'
        for report_id in commits_info_data[x['id']]['report_ids']:
            if report_id not in per_vulnerability_msg:
                per_vulnerability_msg[report_id]=[]
                
            matched_ref_obj = _get_matched_ref_object_by_commit_id(commit_id, spotted_date)
            per_vulnerability_msg[report_id].append(matched_ref_obj)

        if x['patch_prob'][0] >= 0.5 :
            spotted_date = first
        else:
            spotted_date = 'not_spotted'
        for report_id in commits_info_data[x['id']]['report_ids']:
            if report_id not in per_vulnerability_patch:
                per_vulnerability_patch[report_id]=[]
            matched_ref_obj = _get_matched_ref_object_by_commit_id(commit_id, spotted_date)
            per_vulnerability_patch[report_id].append(matched_ref_obj)

        if x['msg_prob'][0]*x['msg_prob'][0] + x['patch_prob'][0]*x['patch_prob'][0] >= 0.25 :
            spotted_date = first
        else:
            spotted_date = 'not_spotted'
        for report_id in commits_info_data[x['id']]['report_ids']:
            if report_id not in per_vulnerability_combined:
                per_vulnerability_combined[report_id]=[]
            matched_ref_obj = _get_matched_ref_object_by_commit_id(commit_id, spotted_date)
            per_vulnerability_combined[report_id].append(matched_ref_obj)


    
    for x in sighted_vulnerabilities:
        data[x]['vfm_message_classification'] = per_vulnerability_msg[x]

    for x in sighted_vulnerabilities:
        data[x]['vfm_patch_classification'] = per_vulnerability_patch[x]

    for x in sighted_vulnerabilities:
        data[x]['vfm_combined_classification'] = per_vulnerability_combined[x]



def main():
    global commits_data
    # Import data on additional commits
    with open(r'src\recalls\data\commit_data_for_rq1.json', 'r') as f:
        commits_data = json.load(f)

        
    global commits_info_data
    # Import data on security commits
    with open(security_relevant_commits_info_file, 'r') as f:
        commits_info_data = json.load(f)

    # Import data from vulnerability databases
    nvd = NvdLoader(r'D:\Projects\VulnerabilityData\new_nvd')
    osv = OsvLoader(r'D:\Projects\VulnerabilityData\new_osv')
    ghsa = OsvLoader(r'D:\Projects\VulnerabilityData\advisory-database\advisories\github-reviewed')
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


    data = {x:{} for x in omni.all_ids}

    # Import creation dates
    load_creation_date(data, omni)

    # Load ecosystems
    load_ecosystem(data, omni)

    # Load repositories
    load_repositories(data, references_data)

    # Import severity
    load_severity(data, omni)

    # Load all refs in new format
    load_all_refs(data, references_data)

    # Load first ref
    load_first_ref(data, references_data)

    # Load first issue
    load_first_issue(data, references_data)

    # Load first commit
    load_first_commit(data, references_data)

    # Load first ref with security phrases
    load_first_sec_phrase(data, references_data)

    # # Most found phrases 
    # global phrases_to_ids
    # asd = sorted({k:len(v) for k,v in phrases_to_ids.items()}.items(), key=lambda x: x[1])

    # Load first issue with security label
    load_first_sec_label(data, references_data)

    # # Load first commit classified as security related (Feature-based)
    load_fb_commit_classification(data)

    # Load first commit classified as security related (own-models)
    # load_dl_commit_classification(data)

    # Load first commit classified as security related (VulFixMiner)
    load_vfm_commit_classification(data)

    with open('recall_analysis.json', 'w') as f:
        json.dump(data, f, default=str)

if __name__ == '__main__':
    main()
    print('ok')


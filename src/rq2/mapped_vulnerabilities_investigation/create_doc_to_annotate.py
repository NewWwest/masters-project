import sys
sys.path.insert(0, r'D:\Projects\aaa')

import json
import pandas as pd
import time
from src.utils.utils import get_files_in_from_directory
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
from src.proxies.GitHubProxy import GithubProxy

output_data_location = 'temp.csv'

input_data_location = 'results/checkpoints_fixMapper'
input_issues_references_dataset = [
    r'D:\Projects\aaa_data\rq2_final_results\most_starred\manually_annotated_mapping.csv',
    r'D:\Projects\aaa_data\rq2_final_results\most_used_npm\manually_annotated_mapping.csv',
    r'D:\Projects\aaa_data\rq2_final_results\most_used_pypi\manually_annotated_mapping.csv',
    r'D:\Projects\aaa_data\rq2_final_results\most_used_mvn\manually_annotated_mapping.csv',
]

def extract_data_from_disclosures():
    nvd = NvdLoader(r'D:\Projects\VulnerabilityData\new_nvd')
    osv = OsvLoader(r'D:\Projects\VulnerabilityData\new_osv')
    ghsa = OsvLoader(r'D:\Projects\VulnerabilityData\advisory-database/advisories/github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)

    res = []
    resx = {}
    for rep in omni.reports:
        refs = omni.references_from_report_list(omni.reports[rep])
        github_refs = [x for x in refs if '/github.com/' in x]
        for gh_ref in github_refs:
            segments = gh_ref.split('#')[0].split('/')
            if len(segments) < 7:
                continue

            repo_full_name = f'{segments[3]}/{segments[4]}'
            data = {
                'repo_owner':segments[3],
                'repo_name':segments[4],
                'cves': omni.related_ids[rep]
            }

            if repo_full_name not in resx:
                resx[repo_full_name] = []

            res.append(data)
            resx[repo_full_name].append(data)


    resxx = {}
    for repo_full_name, items in resx.items():
        sets =set()
        for i in items:
            sets.update(i['cves'])

        resxx[repo_full_name] = list(sets)

    return omni, res, resx, resxx



def main():
    omni, data, data_by_repo, cves_by_repo = extract_data_from_disclosures()
    issues_df = pd.concat([pd.read_csv(x) for x in input_issues_references_dataset])

    res = {}
    all_repos = set()
    matching_repos = set()

    new_df = []
    for i, r in issues_df.iterrows():
        if r['cve_id'].strip() != 'nan':
            new_df.append({
                'issue_url':r['issue_url'],
                'issue_title':r['issue_title'],
                'manual':'ok',
                'cve_id':r['cve_id']
            })
        else:
            segments = r['issue_url'].split('/')
            repo_full_name = f'{segments[4]}/{segments[5]}'
            all_repos.add(repo_full_name)
            if repo_full_name in data_by_repo:
                matching_repos.add(repo_full_name)
                cves =list([x for x in cves_by_repo[repo_full_name] if not x.startswith('CVE')])
                if len(cves) > 0:
                    # add_for_review(new_df, )
                    new_df.append({
                        'issue_url':r['issue_url'],
                        'issue_title':r['issue_title'],
                        'manual':'TODO',
                        'cve_id': []
                    })
                    for cve in cves:
                        title = omni.title_of_report(cve)
                        new_df.append({
                            'issue_url':r['issue_url'],
                            'issue_title':r['issue_title'],
                            'manual':'TODO',
                            'cve_id': cve + ' - ' + title
                        })
                else:
                    new_df.append({
                        'issue_url':r['issue_url'],
                        'issue_title':r['issue_title'],
                        'manual':'ok',
                        'cve_id': 'nan'
                    })
            else:
                new_df.append({
                    'issue_url':r['issue_url'],
                    'issue_title':r['issue_title'],
                    'manual':'ok',
                    'cve_id': 'nan'
                })
            


    df_new = pd.DataFrame(new_df)
    df_new.to_csv(output_data_location, index=None)
    return res

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

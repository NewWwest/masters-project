# TODO: reimplement on new data

# from CommitDownloader import download_commit_data
# from FixMapper import FixMapper
# from GithubService import GithubService
# from OsvReportsLoader import OsvReportsLoader
# import re 
# import time
# import json
# import requests
# from dateutil import parser
# from datetime import tzinfo, timedelta, datetime
# from fuzzywuzzy import fuzz
# from io  import StringIO
# from unidiff import PatchSet
# import matplotlib.pyplot as plt
# import Paths


# api = 'https://api.github.com'
# github_token = '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'
# headers = {'Authorization': f'token {github_token}'}

# def do_get(url):
#     time.sleep(0.4)
#     return requests.get(url, headers=headers)

# def html_to_api_commit(html_uri):
#     try:
#         segments = html_uri.split('/')
#         return f'{api}/repos/{segments[3]}/{segments[4]}/commits/{segments[6]}'
#     except:
#         return None

# def calculate_delays(result_json_path):
#     repo = OsvReportsLoader().load_from_filesystem(path=Paths.GITHUB_ADVISORIES_PATH, save_to_db=False)
    
#     mapper = FixMapper()
#     mapped_commits = mapper.load_file_cache(path='/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/FixPredictor/output_path.json.cache')
#     delays = []
#     analysed_reports = 0
#     for rep in repo.reports:
#         report = repo.reports[rep]
#         if rep in mapped_commits and len(mapped_commits[rep]) > 0:
#             if(len(mapped_commits[rep])>5):
#                 continue

#             print(rep, len(mapped_commits[rep]))
#             analysed_reports +=1
#             for c in mapped_commits[rep]:
#                 api_commit = html_to_api_commit(c)
#                 if api_commit == None:
#                     continue
#                 response = do_get(api_commit)
#                 if response.ok:
#                     data = json.loads(response.text)
#                     committed_date = parser.parse(data['commit']['committer']['date'])
#                     published_date = parser.parse(report.published)
#                     delays.append((published_date-committed_date).days)


# #range=(0,80)
#     with open(result_json_path, 'w') as f:
#         json.dump(delays, f)


# def plot_results(result_json_path):
#     with open(result_json_path, 'r') as f:
#         data = json.load(f)

#     filterrred = [x for x in data if 0<x<400]
#     negative2 = [x for x in data if x<-10]
#     huge = [x for x in data if 400<x]

#     for i in range(len(data)):
#         if data[i] > 400:
#             data[i]=400

#     plt.hist(data, bins = 100, range=(-10,400))
#     print('ploted:', len(filterrred))
#     print('negative:', len(negative2))
#     print('huge:', len(huge))
    
# # allow 5 commits per vulnerability
# # ploted: 454
# # negative: 18
# # huge: 153

# if __name__ == '__main__':
#     # calculate_delays('delays_commits_to_vulnerabilities_5.json')
#     plot_results('delays_commits_to_vulnerabilities_10.json')
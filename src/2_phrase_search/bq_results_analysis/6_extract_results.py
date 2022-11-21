import sys
sys.path.insert(0, r'D:\Projects\aaa')

from src.utils.utils import get_files_in_from_directory
import pandas as pd
import json

input_path = r'src\rq3\bq_results_analysis\results\annotated_longest_phrases.csv'

df = pd.read_csv(input_path)

# for k, g in df.groupby('keyword'):
#     oks = 0
#     vulns = 0
#     for i, x in g.iterrows():
#         if x['manual_rating'] == 'OK':
#             oks += 1
#         elif x['manual_rating'] == 'VULN':
#             vulns += 1
#     print(k, ';', g.shape[0], ';', oks, ';', oks/g.shape[0]*100, ';', vulns, ';', vulns/g.shape[0]*100)


input_path = r'src\rq3\bq_results_analysis\results\annotated_most_entities.csv'
df = pd.read_csv(input_path)

oks = 0
vulns = 0
for k, g in df.iterrows():
    if g['manual_rating'] == 'OK':
        oks += 1
    elif g['manual_rating'] == 'VULN':
        vulns += 1

print('annotated_most_keywords', ';', df.shape[0], ';', oks, ';', oks/df.shape[0], ';', vulns, ';', vulns/df.shape[0])

# input_path = r'src\rq3\bq_results_analysis\results\repos_with_percantages_top_maven.csv'
# df = pd.read_csv(input_path)

# df = df[df.apply(lambda r: r['scanned']>1000,axis=1)]
# df.sort_values('fraction', inplace=True, ascending=False)
# i = 0
# for _, g in df.iterrows():
#     i+=1
#     if i> 20:
#         break
#     print(g['repo_full_name'], ';', g['fraction'], ';', g['with_keywords'], ';', g['scanned'])


# files_annotated = get_files_in_from_directory(r'src\rq3\bq_results_analysis\results', extension='.csv', startswith='anno')
# dfs = [pd.read_csv(f) for f in files_annotated]
# df =pd.concat(dfs)
# df = df.drop_duplicates('html_url')

# oks = 0
# vulns = 0 
# all_counts =0
# for i, x in df.iterrows():
#     all_counts+=1
#     if x['manual_rating'] == 'OK':
#         oks += 1
#     if x['manual_rating'] == 'VULN':
#         vulns += 1
    
# print('final precission', oks, all_counts, oks/all_counts)
# print('final precission', oks+vulns, all_counts, (oks+vulns)/all_counts)
import json
import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import random

input_file = r'results_from_bq_search.csv'
keywords_file = r'src\rq3\extracting_keywords\keywords.csv'


by_keyword = {}
df = pd.read_csv(input_file)
for i, x in df.iterrows():
    entity_url = x['commit_url'] if pd.isna(x['issue_url']) else x['issue_url']
    found_keywords = x['found_keywords'].split(';')
    found_keywords = [k[6:-2] for k in found_keywords]

    for k in found_keywords:
        if k not in by_keyword:
            by_keyword[k] = []
            
        by_keyword[k].append(entity_url)


# finding 1:
# not found keywords:
keywords = pd.read_csv(keywords_file)
not_found_keywords = []
for i, k in keywords.iterrows():
    if k['word'] not in by_keyword:
        not_found_keywords.append(k['word'])

print(len(not_found_keywords), keywords.shape[0], len(not_found_keywords)/keywords.shape[0])

# finding 2:
# smaple most and least popular:
most_popular = {k: v for k, v in sorted(by_keyword.items(), key=lambda item: -len(item[1]))}
with open('temp.json', 'w') as f:
    json.dump(most_popular, f, indent=2)

data_new = []
i = 0
for k,v in most_popular.items():
    i+=1
    if i > 10:
        break
    sample = random.sample(v, min(50, len(v)))
    sample.sort()
    for s in sample:
        html_url = s.replace('api.github.com', 'github.com').replace('/repos/', '/')
        data_new.append({'keyword': k, 'html_url':html_url, 'manual_rating':'XXX', 'url': s})

df_new = pd.DataFrame(data_new)
df_new.to_csv('most_popularxxx.csv', index=False)

data_new = []
longest_found = {k: v for k, v in sorted(by_keyword.items(), key=lambda item: -len(item[0]))}
i = 0
for k,v in longest_found.items():
    i+=1
    if i > 30:
        break
    sample = random.sample(v, min(50, len(v)))
    for s in sample:
        html_url = s.replace('api.github.com', 'github.com').replace('/repos/', '/')
        data_new.append({'keyword': k, 'html_url':html_url, 'manual_rating':'XXX', 'url': s})

df_new = pd.DataFrame(data_new)
df_new.to_csv('longest_foundxxx.csv', index=False)


# print(x['commit_url'])
# print(x['issue_url'])
# print(x['found_keywords'])
# print(x['entity_urls'])
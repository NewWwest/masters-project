import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import random

input_file = r'src\rq3\bq_results_analysis\results_from_bq_search.csv'


by_keyword = {}
df = pd.read_csv(input_file)
df['keyword_count'] = df.apply(lambda row: len(set(row['found_keywords'].split(';'))), axis=1)
df['entities_count'] = df.apply(lambda row: len(set(row['entity_urls'].split(';'))), axis=1)


# finding 1:
# most entities:
data_new = []
most_entities = df.sort_values('entities_count', ascending=False)
i = 0
for _, x in most_entities.iterrows():
    i+=1
    if i > 200:
        break
    
    object_url = x['commit_url'] if pd.isna(x['issue_url']) else x['issue_url']
    html_url = object_url.replace('api.github.com', 'github.com').replace('/repos/', '/')
    found_keywords = x['found_keywords'].split(';')
    found_keywords = [k[6:-2] for k in found_keywords]
    found_keywords = ';'.join(found_keywords)
    data_new.append({'html_url':html_url, 'manual_rating':'XXX', 'keyword': found_keywords, 'url': object_url})
    print(html_url, x['entities_count'])

data_new = sorted(data_new, key=lambda item: item['html_url'])
df_new = pd.DataFrame(data_new)
df_new.to_csv('most_entities.csv', index=False)

# finding 2:
# most keywords:
data_new = []
most_keywords = df.sort_values('keyword_count', ascending=False)
i = 0
for _, x in most_keywords.iterrows():
    i+=1
    if i > 200:
        break
    
    
    object_url = x['commit_url'] if pd.isna(x['issue_url']) else x['issue_url']
    html_url = object_url.replace('api.github.com', 'github.com').replace('/repos/', '/')
    found_keywords = x['found_keywords'].split(';')
    found_keywords = [k[6:-2] for k in found_keywords]
    found_keywords = ';'.join(found_keywords)
    data_new.append({'html_url':html_url, 'manual_rating':'XXX', 'keyword': found_keywords, 'url': object_url})
    print(html_url, x['keyword_count'])

data_new = sorted(data_new, key=lambda item: item['html_url'])
df_new = pd.DataFrame(data_new)
df_new.to_csv('most_keywords.csv', index=False)

# print(x['commit_url'])
# print(x['issue_url'])
# print(x['found_keywords'])
# print(x['entity_urls'])
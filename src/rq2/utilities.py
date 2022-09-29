from wordcloud import WordCloud
from src import secrets
import matplotlib.pyplot as plt
import base64
import json
import time
import requests

headers = {'Authorization': f'token {secrets.github_token}'}


def try_sleep_short():
    time.sleep(0.4)


def try_sleep():
    time.sleep(4)


def do_get(url, long_wait = False):
    if long_wait:
        try_sleep()
    else:
        try_sleep_short()
    return requests.get(url, headers=headers)


def paged_request(endpoint, limit):
    result = []
    page = 1
    while True:
        paged_endpoint = f'{endpoint}&page={page}&per_page=100'
        try_sleep()
        print(f'LOG: Get on {paged_endpoint}')
        x = do_get(paged_endpoint, True)
        data = json.loads(x.text)

        if x.ok:
            if data['total_count'] > 1000:
                print('WARN: Script will not fetch all results. Make the query more specific')
        
            if len(data['items']) == 0:
                break

            for repo in data['items']:
                result.append(repo)
            
            page += 1


            if not data['incomplete_results']:
                break

            if limit != None and len(result) >= limit:
                break
        else:
            print('ERR: handle api errors')
            print(x.text)
            break

    return result

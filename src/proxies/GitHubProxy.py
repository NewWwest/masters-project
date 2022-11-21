
import time
import requests
from src import secrets
from src.utils.constants import github_api, max_allowed_commits_in_link

class GithubProxy:
    headers = {'Authorization': f'token {secrets.github_token}'}

    def __init__(self):
        pass


    def do_get(self, url):
        time.sleep(0.4)  # you can do about 1.4 requests a second
        try:
            response = requests.get(url, headers=self.headers)
            if response.ok:
                return response
            
            if response.status_code == 404 or response.status_code == 410:
                print(f'Resource unfetchable {response.status_code}')
                return None
            else:
                raise Exception(f'Status_Code {response.status_code}')
        except Exception as e:
            print('Exception')
            print(e)
            time.sleep(5)
            response = requests.get(url, headers=self.headers)
            if response.ok:
                return response
            else:
                print(f'Retry failed: {response.status_code}')
                return None


    def get_commit_data(self, repo_owner, repo_name, reference_value):
        api_url = f'{github_api}/repos/{repo_owner}/{repo_name}/commits/{reference_value}'
        response = self.do_get(api_url)
        if response == None:
            return None

        return response.json()


    def get_issue_data(self, repo_owner, repo_name, reference_value):
        api_url = f'{github_api}/repos/{repo_owner}/{repo_name}/issues/{reference_value}'
        response = self.do_get(api_url)
        if response == None:
            return None

        issue = response.json()
        self._get_extra_issue_data(issue)
        self._get_data_as_pr(issue)
        self._get_referenced_data(issue, repo_owner, repo_name)
        return issue


    def _get_extra_issue_data(self, issue_json_obj):
        if 'events_url' in issue_json_obj:
            issue_json_obj['events_data'] = self._simple_paged_request(issue_json_obj['events_url'])
        if 'comments_url' in issue_json_obj:
            issue_json_obj['comments_data'] = self._simple_paged_request(issue_json_obj['comments_url'])
        if 'timeline_url' in issue_json_obj:
            issue_json_obj['timeline_data'] = self._simple_paged_request(issue_json_obj['timeline_url'])
        return issue_json_obj


    def _get_data_as_pr(self, issue_json_obj):
        if 'pull_request' not in issue_json_obj:
            return issue_json_obj
        
        response = self.do_get(issue_json_obj['pull_request']['url'])
        if response == None:
            return issue_json_obj

        pr = response.json()
        issue_json_obj['pull_request_data'] = pr 


        if 'commits' not in pr or (pr['commits'] > 0 and pr['commits'] <= max_allowed_commits_in_link):
            response = self.do_get(pr['_links']['commits']['href'])
            if response != None:
                data = response.json()
                issue_json_obj['pull_request_commits'] = data 
                
        issue_json_obj['pull_request_comments'] = self._simple_paged_request(pr['_links']['review_comments']['href'])

        return issue_json_obj


    def _get_referenced_data(self, issue_json_obj, repo_owner, repo_name):
        repo_full_name = f'{repo_owner}/{repo_name}'
        cross_referenced_issues = [x['source']['issue'] for x in issue_json_obj['timeline_data'] if x['event']=='cross-referenced']
        cross_referenced_events_in_the_same_repo = [x for x in cross_referenced_issues if x['repository']['full_name']==repo_full_name]
        for referenced_issue in cross_referenced_events_in_the_same_repo:
            self._get_extra_issue_data(referenced_issue)
            self._get_data_as_pr(referenced_issue)

        return issue_json_obj


    def get_compare_data(self, repo_owner, repo_name, reference_value):
        api_url = f'{github_api}/repos/{repo_owner}/{repo_name}/compare/{reference_value}'
        response = self.do_get(api_url)
        if response == None:
            return None

        data = response.json()
        return data


    def _simple_paged_request(self, non_query_url):
        per_page = 100
        result = []
        page = 0
        while True:
            page += 1
            url = f'{non_query_url}?per_page={per_page}&page={page}'
            response = self.do_get(url)
            if response != None:
                data = response.json()
                if len(data) > 0:
                    result += data
                if len(data) < per_page:
                    return result
            else:
                return result


    def iterate_search_endpoint(self, endpoint, limit = None):
        result = []
        page = 1
        per_page = 100
        while True:
            paged_endpoint = f'{endpoint}&page={page}&per_page={per_page}'
            time.sleep(2.5)  # Search endpoint has more strict rate limits
            x = self.do_get(paged_endpoint)

            if x != None and x.ok:
                data = x.json()
                if len(data['items']) == 0:
                    break

                result += data['items']
                page += 1


                if not data['incomplete_results'] or len(data) < per_page:
                    break
                if limit != None and len(result) >= limit:
                    break
            else:
                print('Request failed, terminating search')
                break

        return result


    def get_top_contributors(self, repo_owner, repo_name):
        api_url = f'{github_api}/repos/{repo_owner}/{repo_name}/contributors?per_page=100'
        resp = self.do_get(api_url)
        return resp
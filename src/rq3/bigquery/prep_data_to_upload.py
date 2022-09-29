import pandas as pd
keyword_dataset = 'src/rq3/keywords.csv'
repos_datasets = [
    'src/rq3/repos/top_starred_repositories.csv',
    'src/rq3/repos/most_popular_maven_packages.csv',
    'src/rq3/repos/most_popular_npm_packages.csv',
    'src/rq3/repos/most_popular_pypi_packages.csv'
]


def prep_keywords():
    keywords_df = pd.read_csv(keyword_dataset)
    regexes = []
    for k in keywords_df['word']:
        temp = r'(?i)\b'+k+r'\b'
        regexes.append(temp)
    
    new_keywords = pd.DataFrame(regexes)
    new_keywords.to_csv('src/rq3/bigquery/keywords_to_upload.csv', index=False)

def prep_repos():
    dfs = [pd.read_csv(data_file) for data_file in repos_datasets]
    repos = pd.concat(dfs)
    repos_list = set()
    for r in repos['full_name']:
        repos_list.add(r)

    new_keywords = pd.DataFrame(list(repos_list))
    new_keywords.to_csv('src/rq3/bigquery/repos_to_upload.csv', index=False)

if __name__ == '__main__':
    prep_keywords()
    prep_repos()
    
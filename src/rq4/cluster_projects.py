import pandas as pd
from sklearn.cluster import KMeans

input_file = '/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/mined_features_for_most_common_projectsy.csv'
n_clusters = 2
random_state = 42


commits = pd.read_csv(input_file)
features = []
features.append('hours_diff')
features.append('changed_files')
features.append('changed_lines')
features.append('avg_removed_count')
features.append('avg_mod_count')
features.append('avg_added_count')
features.append('dmm_unit_size')
features.append('dmm_unit_complexity')
features.append('dmm_unit_interfacing')
features.append('avg_changed_methods')
features.append('avg_complexity')
features.append('avg_nloc')
features.append('avg_tokens')
features.append('contains_cwe_title')
features.append('contains_security_info')
features.append('contains_security_in_message')

def most_common(lst):
    return max(set(lst), key=lst.count)


def cluster_based_on_most_common_cluster():
    X = commits[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    commits['cluster'] = kmeans.labels_


    labels_per_project = {}
    for index, row in commits.iterrows():
        if row['repo_full_name'] not in labels_per_project:
            labels_per_project[row['repo_full_name']] = []
        labels_per_project[row['repo_full_name']].append(row['cluster'])

    label_assignements = {}
    for x in labels_per_project:
        label_assignements[x] = most_common(labels_per_project[x])

    return label_assignements


def cluster_based_on_mean():
    X = commits.groupby(['repo_full_name'])[features].mean()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    label_assignements = {}
    X['label'] = kmeans.labels_
    for index, row in X.iterrows():
        label_assignements[index]= row['label']
    return label_assignements

def cluster_based_on_mean_diff():
    X = commits.groupby(['repo_full_name'])
    dfs = []
    repos = []
    for i,x in X:
        X1 = x.groupby(['is_vulnerability_fix'])[features].mean()
        asd = {}
        for index, S in X1.iterrows():
            asd[index] = S
        X2 = asd[True]-asd[False]
        repos.append(i)
        dfs.append(X2)
    
    XX = pd.DataFrame(dfs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(XX)
    label_assignements = {}
    for x in range(0, len(repos)):
        label_assignements[repos[x]] = kmeans.labels_[x]
    return label_assignements



# dictx = cluster_based_on_most_common_cluster()
# dictx = cluster_based_on_mean()
dictx = cluster_based_on_mean_diff()

for x in dictx:
    print(x, dictx[x])
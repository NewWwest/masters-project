import sys
sys.path.insert(0, r'D:\Projects\aaa')

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

path_to_file = r'results\security_related_commits_in_vuln.csv'

data = pd.read_csv(path_to_file)

unique_commits = set()
for _, c in data.iterrows():
    unique_commits.add(f'{c["repo_owner"]}/{c["repo_name"]}/{c["commit_sha"]}')

word_could_dict = {}
wordcloud_data = []
for c_id in unique_commits:
    segments = c_id.split('/')
    repo_full_name = f'{segments[0]}/{segments[1]}'
    if repo_full_name not in word_could_dict:
        word_could_dict[repo_full_name] = 0
    word_could_dict[repo_full_name] +=1

print(len(unique_commits))
print(data.shape)

wordcloud_usa = WordCloud(background_color='white',width=6000, height=6000).generate_from_frequencies(word_could_dict)

plt.imshow(wordcloud_usa, interpolation="bilinear")
plt.axis("off")


plt.savefig('aaa.png', bbox_inches='tight', dpi=300)
plt.show()
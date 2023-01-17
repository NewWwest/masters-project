#!/usr/bin/env python3
#
# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------
import dateutil.parser as parser
from datetime import datetime, tzinfo
import pytz
import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv(r'data\rq4_results\features.csv')
repo = 'ether/etherpad-lite'
if repo != None:
    dfx = df[df['label_repo_full_name']==repo]
else:
    dfx=df

print(dfx.shape)
df_sec = dfx[dfx['label_security_related'] == True]
x_axis = dfx['label_commit_date'].apply(parser.parse)
x_axis = [(x.replace(tzinfo=pytz.utc)-datetime.utcnow().replace(tzinfo=pytz.utc)).total_seconds() / 3600 for x in x_axis]
x_axis_sec = df_sec['label_commit_date'].apply(parser.parse)
x_axis_sec = [(x.replace(tzinfo=pytz.utc)-datetime.utcnow().replace(tzinfo=pytz.utc)).total_seconds() / 3600 for x in x_axis_sec]

# columns = X
columns = ['author_to_commiter_date_diff']

for col in columns:
    # plt.title(col)
    plt.scatter(x_axis, dfx[col], color='b', marker='o')
    plt.scatter(x_axis_sec, df_sec[col], color='r', marker='x')
    plt.ylim(-30, 900)
    plt.xlim(-30000,10)
    plt.ylabel('Author to committer date difference [hours]')
    plt.xlabel('Time')
    plt.show()
    print(repo)
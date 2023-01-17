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
import json
import math

cutoff_point = 0.05

data_path = 'data/VC_classification_results.json'
with open(data_path, 'r') as f:
    data = json.load(f)

def comb(a,b):
    return math.sqrt(a*a+b*b)/math.sqrt(2)

for x in data:
    x['combined'] = comb(x['patch_prob'][0], x['msg_prob'][0])
data = sorted([x for x in data if 'commit_size' in x and 'combined' in x], key=lambda x: -x['combined'])

positive_size = sum([x['commit_size'] for x in data if x['is_security_related']==True])
all_size = sum([x['commit_size'] for x in data])

limit = cutoff_point*all_size
accumulated_count = 0
found_positives = 0
for x in data:
    if accumulated_count > limit:
        break

    accumulated_count+=x['commit_size']
    if x['is_security_related'] == True:
        found_positives+=1

all_positives = len([1 for x in data if x['is_security_related']==True])

print(found_positives/all_positives) # cutoff_point = 0.05 => 0.32

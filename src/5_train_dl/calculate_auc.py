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
from sklearn.metrics import  precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import json
import math

data_path = 'data/VC_classification_results.json'

def comb(a,b):
    return math.sqrt(a*a+b*b)/math.sqrt(2)

with open(data_path, 'r') as f:
    data = json.load(f)

y_test = [x['is_security_related'] for x in data]
y_score = [ comb(x['patch_prob'][0], x['msg_prob'][0]) for x in data]

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
auc_precision_recall = auc(recall, precision)
plt.plot(recall, precision)

plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print(auc_precision_recall) # 0.33
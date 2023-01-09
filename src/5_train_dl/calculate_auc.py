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
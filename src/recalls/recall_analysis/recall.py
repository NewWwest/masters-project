import sys
sys.path.insert(0, r'D:\Projects\aaa')

import json
from src.utils.utils import get_files_in_from_directory
import regex as re






hits_rep = {}
hits_word = {}
for x in files:
    print(x)
    with open(x, 'r') as f:
        data = json.load(f)

    for report_id in data:
        report = data[report_id]
        for ref in report:
                print(x)


print(hits_word)
print(hits_rep)

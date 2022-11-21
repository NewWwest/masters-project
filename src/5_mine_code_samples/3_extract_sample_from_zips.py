#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import time
import json
import os
import random
from zipfile import ZipFile

from src.utils.utils import get_files_in_from_directory 

# directory from which to read the zipped results
directory_with_zipped_mining_results = r'zipped-results'

# Configuration of what samples to extract from the mined results
positive_datapoints_count = 250
back_datapoints = 2000
selected_ecosystems = ['mvn'] # ['ALL', 'npm', 'mvn', 'pypi']
selected_miners = ['VulFixMinerMiner'] # ['RollingWindowMiner', 'AddedCodeMiner', 'CodeParserMiner_ast', 'CodeParserMiner_edit', 'VulFixMinerMiner','CommitSizeMiner']
selected_sample_type = ['encoding'] # ['sample', 'encoding']


# directory to which save the extracted sample types
output_directory = 'results/vulfixminer_run1'

npm_code  = set(['js', 'jsx', 'ts', 'tsx', ])
java_code  = set(['java'])
pypi_code = set(['py', 'py3'])


def main(input_location):
    s_selected_ecosystems = set(['ALL', 'npm', 'mvn', 'pypi'])
    s_selected_miners = set(['RollingWindowMiner', 'AddedCodeMiner', 'CodeParserMiner_ast', 'CodeParserMiner_edit', 'VulFixMinerMiner','CommitSizeMiner'])
    s_selected_sample_type = set(['sample', 'encoding'])


    selected_data = []
    zipped_files = get_files_in_from_directory(input_location)
    for file in zipped_files:
        with ZipFile(file, 'r') as zipped:
            for f in zipped.filelist:
                data_commit = json.loads(zipped.read(f.filename))
                ecosystems = ['ALL']
                extensions = set([x['file_name'].split('.')[-1] for x in data_commit])
                if any([x in npm_code for x in extensions]):
                    ecosystems.append('npm')
                if any([x in java_code for x in extensions]):
                    ecosystems.append('mvn')
                if any([x in pypi_code for x in extensions]):
                    ecosystems.append('pypi')

                miners = set([x['sample_type'].split('.')[-1] for x in data_commit])
                sample_type = 'encoding' if '-encodings-' in f.filename else 'sample'

                if sample_type in s_selected_sample_type:
                    if len(miners.intersection(s_selected_miners)) > 0:
                        if len(ecosystems.intersection(s_selected_ecosystems)) > 0:
                            selected_data.append(data_commit)


    positive = []
    back = []
    for x in selected_data:
        if x == None or len(x) == 0:
            continue

        if x[0]['is_security_related']:
            label = 'positive'
        else:
            label = 'background'

        commit_id = x[0]['commit_id'].replace('/','-')
        fn = f'{label}-sample-{commit_id}.json'

        if x[0]['is_security_related']:
            positive.append((fn, x))
        else:
            back.append((fn, x))

    positive_datapoints = random.sample(positive, 225)
    back_datapoints = random.sample(back, 2000)
    for x in positive_datapoints:
        with open(os.path.join(output_directory, x[0]), 'w') as f:
            json.dump(x[1], f)

    for x in back_datapoints:
        with open(os.path.join(output_directory, x[0]), 'w') as f:
            json.dump(x[1], f)
    


if __name__ == '__main__':
    start_time = time.time()
    main(directory_with_zipped_mining_results)
    print("--- %s seconds ---" % (time.time() - start_time))
    
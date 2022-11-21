#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import time
import json
import os
import random
from zipfile import ZipFile


from src.utils.utils import get_files_in_from_directory 

directory_with_zipped_mining_results = r'zipped-results'

valid_extensions = set()
npm_code  = set(['js', 'jsx', 'ts', 'tsx', ])
java_code  = set(['java'])
pypi_code = set(['py', 'py3'])


RollingWindowMiner = 'RollingWindowMiner'
AddedCodeMiner = 'AddedCodeMiner'
CodeParserMiner_ast =  'CodeParserMiner_ast'
CodeParserMiner_edit =  'CodeParserMiner_edit'
VulFixMiner =  'VulFixMinerMiner'
CommitSizeMiner =  'CommitSizeMiner'


def main(input_location):
    ecosystems_all = ['ALL', 'npm', 'mvn', 'pypi']
    miners_all = ['RollingWindowMiner', 'AddedCodeMiner', 'CodeParserMiner_ast', 'CodeParserMiner_edit', 'VulFixMinerMiner','CommitSizeMiner']
    data = {}

    for eco in ecosystems_all:
        data[eco] = {}
        for miner in miners_all:
            data[eco][miner] = {}
            data[eco][miner]['sample'] = []
            data[eco][miner]['encoding'] = []

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
                idx = data_commit[0]['commit_id'] if 'commit_id' in data_commit[0] else data_commit[0]['id']

                for eco in ecosystems:
                    for miner in miners:
                        if '-encodings-' in f.filename:
                            data[eco][miner]['encoding'].append(idx)
                        else:
                            data[eco][miner]['sample'].append(idx)


    for k,v in data.items():
        for kk,vv in v.items():
            for kkk,vvv in vv.items():
                print(k, kk, kkk, len(vvv))


if __name__ == '__main__':
    start_time = time.time()
    main(directory_with_zipped_mining_results)
    print("--- %s seconds ---" % (time.time() - start_time))
    
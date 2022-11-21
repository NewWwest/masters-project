#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json
from src.utils.utils import get_files_in_from_directory 


class NvdLoader:
    def __init__(self, path: str, start_year = 2000) -> None:
        self.reports = {}
        self.report_ids = set()
        self.data_path = path
        self.start_year = start_year
        self._load()


    def _load(self):
        all_advisories = get_files_in_from_directory(self.data_path, extension='.json')
        for advisory_file in all_advisories:
            self._load_file(advisory_file)

    def _load_file(self, file):
        with open(file, encoding='utf-8') as f:
            d = json.load(f)
            for report in d['CVE_Items']:
                cve_id = report['cve']['CVE_data_meta']['ID']
                self.reports[cve_id] = report
                self.report_ids.add(cve_id)


if __name__ == '__main__':
    path_to_nvd_dump_json_files = 'path_to_nvd_dump_json_files'
    repo = NvdLoader(path_to_nvd_dump_json_files)
    print(len(repo.report_ids))
    # Since 2002 => 193404
    # Since 2010 => 151699

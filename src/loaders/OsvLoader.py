#!/usr/bin/env python3
# import sys
# sys.path.insert(0, r'PATH_TO_REPO')

import json

from src.utils.utils import get_files_in_from_directory 

class OsvLoader:
    def __init__(self, path) -> None:
        self.data_path = path
        self.reports = {}
        self.reports_also_by_aliases = {}
        self._load()


    def _load(self):
        files = get_files_in_from_directory(self.data_path, extension='.json')
        self._read_reports(files)
        self.alphabetical_order = list(self.reports)
        self.alphabetical_order.sort()


    def _read_reports(self, files):
        for file in files:
            with open(file, encoding="utf8") as f:
                osv_report = json.load(f)
                id = osv_report['id']
                self.reports[id] = osv_report

                if id not in self.reports_also_by_aliases:
                    self.reports_also_by_aliases[id]=[]
                self.reports_also_by_aliases[id].append(osv_report)

                if 'aliases' in osv_report:
                    for alias in osv_report['aliases']:
                        if alias not in self.reports_also_by_aliases:
                            self.reports_also_by_aliases[alias]=[]
                        self.reports_also_by_aliases[alias].append(osv_report)


    def get_id_aliases(self, report_id):
        related_ids = [report_id]
        if report_id in self.reports and 'aliases' in self.reports[report_id]:
            related_ids += self.reports[report_id]['aliases']


        if report_id in self.reports_also_by_aliases:
            for x in self.reports_also_by_aliases[report_id]:
                if 'aliases' in x and report_id in x['aliases']:
                    related_ids.append(x['id'])

        return list(set(related_ids))


if __name__ == '__main__':
    path_to_osv_data_dump = 'path_to_osv_data_dump'
    repo = OsvLoader(path_to_osv_data_dump)
    print(len(repo.reports))
    # OSV => 30387
    # GHSA => 8696
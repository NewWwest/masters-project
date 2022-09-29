import os
import json

from src.utils.constants import path_separator

focus_ecosystems = ['PyPI', 'Maven', 'npm']

class OsvLoader:
    def __init__(self, path, ecosystems = None) -> None:
        self.data_path = path
        self.reports = {}
        self.reports_also_by_aliases = {}
        self.ecosystems_to_load = set(ecosystems) if ecosystems != None else None
        self._load()


    def _load(self):
        files = self._enumerate_files(self.data_path)
        self._read_reports(files)
        self.alphabetical_order = list(self.reports)
        self.alphabetical_order.sort()


    # TODO: use the one from utils
    def _enumerate_files(self, rootdir):
        all_advisories = []
        for root, subdirs, files in os.walk(rootdir):
            ecosystem = root.split(path_separator)[-1]
            if self.ecosystems_to_load != None and ecosystem not in self.ecosystems_to_load:
                continue
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    all_advisories.append(file_path)

        return all_advisories


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
    repo = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    print(len(repo.reports))
    # All => 30387
    # npm+pypi+mvn => 7922
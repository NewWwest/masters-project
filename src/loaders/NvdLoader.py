import os
import json

class NvdLoader:
    def __init__(self, path: str, start_year = 2000) -> None:
        self.reports = {}
        self.report_ids = set()
        self.data_path = path
        self.start_year = start_year
        self._load()


    def _load(self):
        all_advisories = self._enumerate_files(self.data_path)
        for advisory_file in all_advisories:
            self._load_file(advisory_file)


    # TODO: use the one from utils
    def _enumerate_files(self, path):
        all_advisories = []
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)) and file.endswith('.json'):
                year_in_file  = file[-9:-5]
                if int(year_in_file) < self.start_year:
                    continue

                file_path = os.path.join(path, file)
                all_advisories.append(file_path)

        return all_advisories


    def _load_file(self, file):
        with open(file, encoding='utf-8') as f:
            d = json.load(f)
            for report in d['CVE_Items']:
                cve_id = report['cve']['CVE_data_meta']['ID']
                self.reports[cve_id] = report
                self.report_ids.add(cve_id)




if __name__ == '__main__':
    repo = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    print(len(repo.report_ids))
    # Since 2002 => 193404
    # Since 2010 => 151699


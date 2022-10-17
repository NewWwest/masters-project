from typing import Iterable
import dateutil.parser as parser
from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader


class DualReport:
    NVD_type = 1
    OSV_type = 2
    def __init__(self, report_source, report) -> None:
        self.report_source = report_source # 1 for NVD, 2 FOR OSV
        self.report = report


class OmniLoader:
    def __init__(self, nvdLoader: NvdLoader, osvLoader: OsvLoader, ghsaLoader: OsvLoader) -> None:
        self._nvdLoader = nvdLoader
        self._osvLoader = osvLoader
        self._ghsaLoader = ghsaLoader
        self.all_ids = set()
        self.reports = {}
        self.related_ids = {}
        self._import_related_ids()
        self._remap_reports()


    def _remap_reports(self):
        for x in self.all_ids:
            self.reports[x] = self._reports_by_report_id(x)


    def _import_related_ids(self):
        for nvd_report_id in self._nvdLoader.reports:
            self.all_ids.add(nvd_report_id)
            self._set_related_ids(nvd_report_id)
        self._load_osv_ids(self._osvLoader)
        self._load_osv_ids(self._ghsaLoader)


    def _load_osv_ids(self, loader):
        for osv_report_id in loader.reports:
            self.all_ids.add(osv_report_id)
            self._set_related_ids(osv_report_id)
            if 'aliases' in loader.reports[osv_report_id]:
                for alias in loader.reports[osv_report_id]['aliases']:
                    self._set_related_ids(alias)
                    self.all_ids.add(alias)


    def _set_related_ids(self, report_id):
        if report_id in self.related_ids:
            return 

        related_ids = []
        related_ids += self._osvLoader.get_id_aliases(report_id)
        related_ids += self._ghsaLoader.get_id_aliases(report_id)
        self.related_ids[report_id] = list(set(related_ids))


    def _reports_by_report_id(self, report_id):
        related_ids = self.related_ids[report_id]
        reports = []
        for related_id in related_ids:
            if related_id in self._nvdLoader.reports:
                temp = DualReport(DualReport.NVD_type, self._nvdLoader.reports[related_id])
                reports.append(temp)
            if related_id in self._osvLoader.reports_also_by_aliases:
                temp = DualReport(DualReport.OSV_type, self._osvLoader.reports_also_by_aliases[related_id])
                reports.append(temp)
            if related_id in self._ghsaLoader.reports_also_by_aliases:
                temp = DualReport(DualReport.OSV_type, self._ghsaLoader.reports_also_by_aliases[related_id])
                reports.append(temp)

        return reports

        
    def title_of_report(self, report_id):
        if report_id in self._nvdLoader.reports:
            #TODO: this line was never used
            return self._nvdLoader.reports[report_id]['cve']['description']['description_data'][0]['value']
        if report_id in self._osvLoader.reports:
            if 'summary' in self._osvLoader.reports[report_id]:
                return self._osvLoader.reports[report_id]['summary']
            elif 'details' in self._osvLoader.reports[report_id]:
                return self._osvLoader.reports[report_id]['details']
            else:
                return report_id
        if report_id in self._ghsaLoader.reports:
            if 'summary' in self._ghsaLoader.reports[report_id]:
                return self._ghsaLoader.reports[report_id]['summary']
            elif 'details' in self._ghsaLoader.reports[report_id]:
                return self._ghsaLoader.reports[report_id]['details']
            else:
                return report_id


    def references_from_report_list(self, report_list:Iterable[DualReport]):
        references = []
        for report in report_list:
            if report.report_source == DualReport.OSV_type:
                if 'references' in report.report:
                    references += [x['url'] for x in report.report['references']]
            elif report.report_source == DualReport.NVD_type:
                if 'references' in report.report['cve'] and 'reference_data' in report.report['cve']['references']:
                    references += [x['url'] for x in report.report['cve']['references']['reference_data']]
            else:
                raise Exception(f'Invalid report source {report.report_source}')

        return list(set([x.lower() for x in references]))

    def publish_date_of_report(self, report_id):
        if report_id not in self.related_ids:
            return None

        ids = self.related_ids[report_id]
        dates = []
        for id in ids:
            if id in self._nvdLoader.reports:
                res = {
                    'report_id':id,
                    'publish_date':parser.parse(self._nvdLoader.reports[id]['publishedDate'])
                }
                dates.append(res)
            if id in self._osvLoader.reports:
                res = {
                    'report_id':id,
                    'publish_date':parser.parse(self._osvLoader.reports[id]['published'])
                }
                dates.append(res)
            if id in self._ghsaLoader.reports:
                res = {
                    'report_id':id,
                    'publish_date':parser.parse(self._ghsaLoader.reports[id]['published'])
                }
                dates.append(res)

        return dates

    
    def ecosystems_of_a_report(self, report_id):
        if report_id not in self.related_ids:
            return None

        ids = self.related_ids[report_id]
        ecosystems = []
        for id in ids:
            if id in self._nvdLoader.reports:
                cpes = []
                for n in self._nvdLoader.reports[id]['configurations']['nodes']:
                    cpes += self._dfs_nodesearch(n)

                xxxx = set([cpe.split(':')[10] for cpe in cpes])
                xxxx.discard('*')
                xxxx.discard('-')
                ecosystems += list(xxxx)
            if id in self._osvLoader.reports:
                ec = [x['package']['ecosystem'] for x in self._osvLoader.reports[id]['affected']]
                ecosystems += ec 
            if id in self._ghsaLoader.reports:
                ec = [x['package']['ecosystem'] for x in self._ghsaLoader.reports[id]['affected']]
                ecosystems += ec 

        return ecosystems

    def _dfs_nodesearch(self, node):
        cpes = [c['cpe23Uri'] for c in node['cpe_match'] if c['vulnerable']]
        for child in node['children']:
            cpes+=self._dfs_nodesearch(child)
        return cpes

                


if __name__ == '__main__':
    nvd = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    osv = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    ghsa = OsvLoader('/Users/awestfalewicz/Private/data/advisory-database/advisories/github-reviewed')
    omni = OmniLoader(nvd, osv, ghsa)
    print(len(nvd.reports))
    print(len(osv.reports))
    print(len(ghsa.reports))
    print(len(omni.reports))


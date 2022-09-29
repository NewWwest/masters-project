from src.src2.loaders.NvdLoader import NvdLoader
from src.src2.loaders.OsvLoader import OsvLoader

def main():
    _nvdLoader = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    _osvLoader = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')

    nvd_cve_all = _nvdLoader.report_ids.copy()
    osv_cve_aliases_arr = [_osvLoader.reports[x]['aliases'] for x in _osvLoader.reports if 'aliases' in _osvLoader.reports[x]]
    osv_cve_aliases_x = [item for sublist in osv_cve_aliases_arr for item in sublist]
    osv_cve_aliases_x = set([alias for alias in osv_cve_aliases_x if alias.startswith('CVE')])
    osv_cve_all = osv_cve_aliases_x

    print('Vulnerabilities in:')
    print('NVD', len(nvd_cve_all))
    print('OSV', len(_osvLoader.report_ids))

    print('CVE-IDS in:')
    print('NVD', len(nvd_cve_all))
    print('OSV', len(osv_cve_all))

    print('Cross sections:')
    print('CVE-IDs in OSV and in NVD', len(osv_cve_all.intersection(nvd_cve_all)))
    print('CVE-IDs in OSV and not in NVD', len(osv_cve_all.difference(nvd_cve_all)))
    print('CVE-IDs not in OSV and in NVD', len(nvd_cve_all.difference(osv_cve_all)))


if __name__ == '__main__':
    main()

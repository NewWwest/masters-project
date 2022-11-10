# import sys
# sys.path.insert(0, PATH_TO_REPO)


from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
import matplotlib.pyplot as plt

path_to_osv_dump = ''
path_to_nvd_dump = ''
path_to_ghsa_dump = ''

def main():
    nvd = NvdLoader(path_to_nvd_dump)
    osv = OsvLoader(path_to_osv_dump)
    ghsa = OsvLoader(path_to_ghsa_dump)
    omni = OmniLoader(nvd, osv, ghsa)

    vuln_sets = set()
    first_dates = {}

    for x in omni.related_ids:
        related = omni.related_ids[x]
        related = list(related)
        related.sort()
        key = ';'.join(related)
        vuln_sets.add(key)

    print(len(nvd.reports))
    print(len(osv.reports))
    print(len(ghsa.reports))
    print(len(omni.reports))
    print(len(vuln_sets))     

    for x in vuln_sets:
        report_ids = x.split(';')
        asd = []
        for r in report_ids:
            asd += omni.publish_date_of_report(r)

        asd = [x['publish_date'] for x in asd]
        first = min(asd)
        if first.year >= 2000:
            if first.year not in first_dates:
                first_dates[first.year] = 0
            
            first_dates[first.year] +=1


    sorted_years = list(first_dates.keys())
    sorted_years.sort()
    values_per_year = [first_dates[s] for s in sorted_years]
    

    plt.bar(sorted_years, values_per_year, width=1.0)
    plt.xticks(sorted_years, sorted_years, rotation=90)
    plt.xlim([sorted_years[0]-0.5, sorted_years[-1]+0.5])
    plt.ylabel("Number of Vulnerabilities")
    plt.xlabel("Year")

    plt.show()



if __name__ == '__main__':
    main()
    print('ok')  
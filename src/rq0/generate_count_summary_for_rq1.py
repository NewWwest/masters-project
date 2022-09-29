import sys
sys.path.insert(0, r'D:\Projects\aaa')


from src.loaders.NvdLoader import NvdLoader
from src.loaders.OsvLoader import OsvLoader
from src.loaders.OmniLoader import OmniLoader
import matplotlib.pyplot as plt
from wordcloud import WordCloud

commits = set()
commits_data = {}
input_data_location = 'results/checkpoints_fixMapper'
keywords_regexes_path = r'src\rq3\bigquery\keywords_to_upload.csv'
all_labels = []




def main():
    nvd = NvdLoader(r'D:\Projects\VulnerabilityData\new_nvd')
    osv = OsvLoader(r'D:\Projects\VulnerabilityData\new_osv')
    ghsa = OsvLoader(r'D:\Projects\VulnerabilityData\advisory-database/advisories/github-reviewed')
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
from src.src2.loaders.NvdLoader import NvdLoader
from src.src2.loaders.OsvLoader import OsvLoader
from src.src2.loaders.OmniLoader import OmniLoader

if __name__ == '__main__':
    nvd = NvdLoader('/Users/awestfalewicz/Private/data/new_nvd')
    osv = OsvLoader('/Users/awestfalewicz/Private/data/new_osv')
    omni = OmniLoader(nvd, osv)

    all=0
    commits=0
    commit=0
    issues=0
    issue=0
    pull=0
    compare = 0
    diff = 0
    blob=0
    tree=0
    releases=0
    advisories=0
    wiki=0
    too_short=0
    other=0

    for report_id in omni.reports:
        refs = omni.references_from_report_list(omni.reports[report_id])
        github_ref = [x for x in refs if '/github.com/' in x]
        for url in github_ref:
            all+=1
            segments = url.split('/')

            if len(segments) < 7:
                too_short+=1
            elif '/commit/' in url:
                commit+=1
            elif '/commits/' in url:
                commits+=1
            elif '/issue/' in url:
                issue+=1
            elif '/issues/' in url:
                issues+=1
            elif '/pull/' in url:
                pull+=1
            elif '/compare/' in url:
                compare+=1
            elif '/diff/' in url:
                diff+=1
            elif '/blob/' in url:
                blob+=1
            elif '/tree/' in url:
                tree+=1
            elif '/releases/' in url:
                releases+=1
            elif '/security/advisories' in url or 'github.com/advisories' in url:
                advisories+=1
            elif 'wiki' in url:
                wiki+=1
            else:
                other+=1

    print('all', all)
    print('commits',commits)
    print('commit',commit)
    print('issues',issues)
    print('issue',issue)
    print('pull',pull)
    print('compare',compare)
    print('diff',diff)
    print('blob',blob)
    print('tree',tree)
    print('releases',releases)
    print('advisories',advisories)
    print('wiki', wiki)
    print('too_short', too_short)
    print('other', other)
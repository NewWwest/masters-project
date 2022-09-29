import matplotlib.pyplot as plt
import pandas as pd

ymax_ = 25
bins_ = range(-90, 1070, 30)
ymaxx_ = 8
binsx_ = range(-5, 50, 1)

paths = [
    'src/src2/rq2/all_vulnerabilities_sec_issues.csv',
    'src/src2/rq2/npm_vulnerabilities_sec_issues.csv',
    'src/src2/rq2/mvn_vulnerabilities_sec_issues.csv',
    'src/src2/rq2/pypi_vulnerabilities_sec_issues.csv'
    ]

fig, axd = plt.subplot_mosaic([['top', 'top','top'],['left','mid','right'],['leftx','midx','rightx']],
                              constrained_layout=True)

df = pd.read_csv(paths[0])
axd['top'].hist(df['diff'],  bins=bins_)
axd['top'].axis(ymin=0)
axd['top'].set_ylabel('Number of issues')
axd['top'].set_xlabel('Delay in days')
axd['top'].set_title('All vulnerabilities')



df = pd.read_csv(paths[1])
axd['left'].hist(df['diff'],  bins=bins_)
axd['left'].axis(ymin=0, ymax=ymax_)
axd['left'].set_ylabel('Number of issues')
axd['left'].set_xlabel('Delay in days')
axd['left'].set_title('npm vulnerabilities')

df = pd.read_csv(paths[2])
axd['mid'].hist(df['diff'],  bins=bins_)
axd['mid'].axis(ymin=0, ymax=ymax_)
axd['mid'].set_ylabel('Number of issues')
axd['mid'].set_xlabel('Delay in days')
axd['mid'].set_title('Maven vulnerabilities')

df = pd.read_csv(paths[3])
axd['right'].hist(df['diff'],  bins=bins_)
axd['right'].axis(ymin=0, ymax=ymax_)
axd['right'].set_ylabel('Number of issues')
axd['right'].set_xlabel('Delay in days')
axd['right'].set_title('PyPI vulnerabilities')



df = pd.read_csv(paths[1])
axd['leftx'].hist(df['diff'],  bins=binsx_)
axd['leftx'].axis(ymin=0, ymax=ymaxx_)
axd['leftx'].set_ylabel('Number of issues')
axd['leftx'].set_xlabel('Delay in days')
axd['leftx'].set_title('npm vulnerabilities')

df = pd.read_csv(paths[2])
axd['midx'].hist(df['diff'],  bins=binsx_)
axd['midx'].axis(ymin=0, ymax=ymaxx_)
axd['midx'].set_ylabel('Number of issues')
axd['midx'].set_xlabel('Delay in days')
axd['midx'].set_title('Maven vulnerabilities')

df = pd.read_csv(paths[3])
axd['rightx'].hist(df['diff'],  bins=binsx_)
axd['rightx'].axis(ymin=0, ymax=ymaxx_)
axd['rightx'].set_ylabel('Number of issues')
axd['rightx'].set_xlabel('Delay in days')
axd['rightx'].set_title('PyPI vulnerabilities')

plt.show()
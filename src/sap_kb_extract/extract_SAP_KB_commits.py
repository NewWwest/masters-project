import yaml
import pandas as pd
from src.utils.utils import get_files_in_from_directory

sap_kb_repo = 'C:\\Projects\\project-kb\\statements'
output_file = 'sap-extracted-commits.csv'

mapping = {
    'https://gitbox.apache.org/repos/asf/hive.git':'https://github.com/apache/hive',
    'https://android.googlesource.com/platform/frameworks/base.git':'https://github.com/aosp-mirror/platform_frameworks_base',
    'https://gitbox.apache.org/repos/asf/cordova-plugin-file-transfer.git':'https://github.com/apache/cordova-plugin-file-transfer',
    'https://gitbox.apache.org/repos/asf/cxf-fediz.git':'https://github.com/apache/cxf-fediz',
    'https://gitbox.apache.org/repos/asf/struts.git':'https://github.com/apache/struts',
    'https://git-wip-us.apache.org/repos/asf/activemq-apollo.git':'https://github.com/apache/activemq-apollo',
    'https://git-wip-us.apache.org/repos/asf/camel.git':'https://github.com/apache/camel',
    'https://gitbox.apache.org/repos/asf/ambari.git':'https://github.com/apache/ambari',
    'https://android.googlesource.com/platform/packages/apps/Exchange.git':None, #no_mirror
    'https://git-wip-us.apache.org/repos/asf/cxf.git':'https://github.com/apache/cxf',
    'https://git-wip-us.apache.org/repos/asf/lucene-solr':'https://github.com/apache/lucene-solr',
    'https://git-wip-us.apache.org/repos/asf/tapestry-5.git.git':'https://github.com/apache/tapestry-5',
    'https://gitbox.apache.org/repos/asf/hbase.git':'https://github.com/apache/hbase',
    'https://git-wip-us.apache.org/repos/asf/activemq.git':'https://github.com/apache/activemq',
    'https://git-wip-us.apache.org/repos/asf/tomee.git':'https://github.com/apache/tomee',
    'https://gitbox.apache.org/repos/asf/activemq.git':'https://github.com/apache/activemq',
    'https://android.googlesource.com/platform/frameworks/opt/telephony.git':None, #no_mirror
    'https://gitbox.apache.org/repos/asf/tapestry-5':'https://github.com/apache/tapestry-5',
    'https://gitbox.apache.org/repos/asf/cxf.git':'https://github.com/apache/cxf',
    'https://android.googlesource.com/platform//external/conscrypt.git':None, #no_mirror
} # in total 4 commits left out

parsed_data = {}
data = get_files_in_from_directory(sap_kb_repo, extension='.yaml')
for d in data:
    with open(d, "r", encoding='UTF-8') as f:
        dict = yaml.safe_load(f)
        if 'fixes' in dict:
            for fix in dict['fixes']:
                if 'commits' in fix:
                    for c in fix['commits']:
                        if '/github.com/' in c['repository']:
                            link = c['repository']
                        elif c['repository'] in mapping and mapping[c['repository']]:
                                link = mapping[c['repository']]
                        else:
                            continue
                        if link not in parsed_data:
                            parsed_data[link]=[]
                            
                        parsed_data[link].append({
                            'report_id':dict['vulnerability_id'],
                            'sha': c['id'], 
                            'repo_url': link
                        })
    
datax = []
for k,v in parsed_data.items():
    segments = k.split('/')
    for cx in v:
        print(len(cx['sha']))
        datax.append({
            'repo_owner': segments[3],
            'repo_name': segments[4],
            'sha':cx['sha']
        })    

data_df = pd.DataFrame(datax)
data_df.to_csv(output_file, index=False)



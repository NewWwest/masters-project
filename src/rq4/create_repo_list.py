
from src.rq4.CommitSelector import CommitSelector
from src.rq4.FeatureCalculator import FeatureCalculator
from src.rq4.CommitMiner import CommitMiner
from src.rq4.CommitPreselector import CommitPreselector
from src.rq4.RepoDownloader import RepoDownloader
from transformers import AutoTokenizer
import json


cps = CommitPreselector()
cps.filter_commits_for_repo('moment/moment')
fc = FeatureCalculator()
fc.process_repo('moment/moment')
cs = CommitSelector('/Users/awestfalewicz/Projects/2022-internship-andrzej-westfalewicz/fake_mapping.json')
cs.select_commits_for_repo('moment/moment')



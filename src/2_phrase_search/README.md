# Phrase Search

This directory contains code and indtermediate results of the phrase search experiment. 

## Extracting keywords

The first step of the process is the extraction of keyword from the code. For that the `analyze_patches`. Given a set of extracted patches from security related and background commits it will extract tokens for each of the patches (`mine_tokens` function). The using the `rank_tokens` one can produce sets of tokens ranked accoring to one of the ratios calculated in the function. The produced listing of keyword can be then manually review to extract the non-general, non-random keywords.


## Keywords

The keywords directory contains the sets of keyword used in the phrase-search including the base datasets and the intermediate stages. First the `original` base datasets are the set of phrases by Le et al. and the CWE lists as download from NVD. Next, in `to_review` there are datasets of the deduplicated keywords and the CWE tiles not covered by any phrase. These are manually reviewed to extract phrases from as many CWE as possible and reject the invalid ones. Finally, to the dataset `after_review` we add the tokens mined from code and personally chosen keywords to form the `final` dataset.


## Big Query

The big query directory contains the necessary data to reproduce the results of the text search done on Google Cloud Big Query. First the keywords dataset contain the phrases used in search represented as regexes. Then, the repos set contains the concatenated set of most popular packages and the most starred repositories. Finnaly, the query is also provided. The query requires 4 strings to be replaced: 
- {GHA_TABLE} pointing to the specific table of the GitHub Archive e.g. `githubarchive.day.20190101`
- {REPOS_TABLE} - The table which contains the uploaded repositories
- {KEYWORDS_TABLE} - The table which contains the uploaded regex formed search phrases
- {RESULT_TABLE} - the name of the table that should be created as the result of the search

In the query first a multi-union of subqueries is executed that extract the relevant natural-language fields from the specified repositories. Then, the results of the union is LEFT-JOINed to the table of regex to detect the phrases we are looking for. Finally this results is used as the parameter to the CREATE table clause.


## BQ results analysis

Finally the results analysis contains the stages for processing the results and sampling the entities reviewd in the thesis.

### Concat

The first script reads the compressed files from the data dump produced by BigQuery and concatenates them into one single CSV file. The results that had security phrases found in them are additionally separately.

### Deduplicate

Next the results of the query have to be deduplicated. First, the same main entity (issue or commit) can be found in different sets =, for example when the issues was active for multiple months ion 2022. Second, the found keywords of each main entity are concatinated into a list.

### Analyze & Extract

The following scripts use the results prepared by those above to sample data to be reviewed and process the reviewed objects.
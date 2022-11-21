# Mine Code

This directory contains code related to running the pipeline for extract code samples from security-related and background commits. It requires the CSV file containing the commits to mine as produced in 1_extracting_references.

## Mine code probabilistic

This file runs the mining and produces the samples and the tokenized encodings of the commits.
For the full functionality the following external dependencies have to be installed.
- Available from apt-get
    - openjdk-17-jdk (for `CodeParserMiner`)
    - openjdk-17-jre  (for `CodeParserMiner`)
    - python3-pip
    - maven (for `CodeParserMiner`)
    - git
    - unzip
    - libarchive13 (for `CodeParserMiner`)
    - npm (for `CodeParserMiner`)
- Available from pip3
    - google-cloud-storage 
    - pydriller==1.15.1 
    - regex 
    - pandas 
    - transformers 
    - parso (for `CodeParserMiner`)
- To be installed manually:
    - https://www.srcml.org/ (for `CodeParserMiner`)
    - https://github.com/GumTreeDiff/pythonparser (for `CodeParserMiner`)
    - https://github.com/GumTreeDiff/jsparser (for `CodeParserMiner`)

Next, configure the variables at the top of the file to set the output, and work directories and the handled extensions. As noted in the file, change also the configuration in `GumTreeProxy` to minic the setup of the CodeParser server. The CodeParser is a interface to the `GumTreeDiff` tool utilised by the `CodeParserMiner`. When needed, the miner will saved the old and the new version of the file in the `workdir` and call the CodeParser server to process the files. Two json files are produced, one with the AST of the new commit and one with the edit script between the files. Those are then loaded back into the miner and removed. 

Next, configure the miners that will be used in the mining process. If a miner is not required it can be commented out from the `mine_a_repo` function and not provided in the collection to the combined miner.

Finally, if you don't want to zip or upload the results, remove the injection of the `uploader`. If the `uploader` is provided, after a repository is mined, all produced files will be aggregated in one zip and the individual files will be removed. Additionally, if used with the GcProxy, the file will be uploaded to specified bucket.


### Result

The result is a collection of files with each file containing the results of a miner for a commit. 
The filename is structured like: '{directory_of_the_miner}/{positive/background}-{samples/encodings}-{owner}-{repo}-{sha}.json', where:
- directory is the dicrectory configured for each miner,
- positive/background indicates whenever the commit was provided as security-related,
- samples/encodings shows whenever the file contains the sampled code or the encodding ready to be processed by the DL models,
- owner, repo, sha identify the commit.

Each file is a collection of mined samples which slightly depend on the used miner. Ususally the ouput contains
- commit_id (or id): owner/repo/sha
- sample_type: the name of the miner
- file_name: the path to the modified filen sample was extracted from
- is_security_related: True or False
- commit_title: a cropped commit message
- commit_sample: the extracted sample of the code
- commit_size: the number of changed lines in the commit (only for `CommitSizeMiner`)


## Investigate zipped results

A script containing code that can be used to investigate the amounts of mined code. It displays how many commits were mined for each ecosystem, miner, positive/background and samples/encodings.


## Extract sample from zips

A script that can be used to extract the specific dataset from the mined code. For example, to extract only the encodings mined woth the AST method for the Java language.
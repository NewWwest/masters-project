# Labeled issues investigation

This dierectory contains the investigation into the security labeled issues preseted in Experiment 2 of the thesis.

The main file of the directory  is the `process_repositories.py` file which was run for every ecosystem as discussed in the thesis. It requires the configuration of directories, Github token from secrets and the input repositories to process. The output of script will be various files of issues at each stage with different information each. The file `extract_security_relevant_commits` can be used  to extract commits linked to issues. Next, files in `mapped_vulnerabilities_investigation` contain code that helps to generate to review manually and map the issues to vulnerabilities.
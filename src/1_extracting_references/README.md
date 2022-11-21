# Extracting references

This directory contians the first steps in the pipeline of utilising vulnerability data, which are extracting the references and resolving them against the GitHub API.

## Investigate Different Github Url Types

By running this file you can get insights on what github references are available in the vulnerabilities. Keep in mind that many duplicates are present at this stage.

## Mine Github References

This script can be used to extract references from vulnerabilities in a standarized way that allows to remove duplicates. Furthermore, in the second stage the references are resolved against the GitHub API. Note, that this process is very time consuming because of the volume of the data and because requesting information from the API is subjected to rate limits.

The resulting file (files) is a dictionary of collections of references for each report id. Each element in every collection contains the report id, aliases of that report, the repository owner and name, the type and the identifier of the reference and the fetched data of the reference.


## Extract Security Relevant Commits

Use this script to extract commits from the results produced by the script above. Commits are extract from the entities as decribed in the thesis: commits are copied, compares are converted to a list of commits and linked commits are extracted from issues.
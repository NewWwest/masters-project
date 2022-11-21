# Loaders

The loaders classes are used to import the raw data dumps from NVD, OSV and GHSA. Use the OmniLoader as the single point of extracting the important data from vulnerabilies. 

The OmniLoader requires all 3 vulnerability sources - NVD, OSV and GHSA, but it can be easily modified to be used without any of them. Use the `reports` field in the OmniLoader to iterate the vulnerabilities. It contains a collection of related vulnerabilities for each report id (e.g. CVE, GHSA, DLA), including the report itself. Then, proceed to use the required function by providing the report id or the collection of reports to extract relevant data. 
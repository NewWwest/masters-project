# Mining

This directory contains classes implementing the mining methods used in the project.

## Code Features Miners

The `CodeFeaturesMiners` directory contains the code used to mine features in `4_mine_features`. In particular the `CommitMiner` is the entry point for the 'module'. It iterates over the commits in the specified repository and mines Commit- and File-level features. The `constants_and_configs` contains the configuration used by the miners, including which extensions map to which ecosystems.


## CodeMiners

The `CodeFeaturesMiners` directory contains the code used to mine code saples in `5_mine_code_samples`. The main class in the directory is the `CombinedMiner` which iterates over commits in the repository and calls each of the miners from the collection injected into it.

### CodeParser

CodeParser is a mini project in Maven to run the GumTreeDiff tool. The `CodeParserMiner` saves files to be analyzed to the disk and the pings the server to process these files. The `GumTreeDiffParserProxy.java` file contains logic to load and process these files and save the result as json file back to the disk. The miner then loads the result and removes the temporary files.

## Issues Miners

TODO


## Vulnerability Miners

Contains only one class `GitHubReferencesMiner`, which scans the vulnerabilities for references to GitHub and resolves them against the GitHub API using `GitHubProxy`. 
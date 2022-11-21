# Mine Features

To run the mining pipeline for the commit features one requires the security-related related commits as mined in 1_extracting_references (or equivalent). Put the path to the csv with the commits in the security_commits_file variable. Then, configure the folders where temporary and final results are saved.

## Get contributors

The first step in mining features is fetching the contributors for the repositories. This is used in one of the author-based features as we checking if the user making the commit is an 'established' maintainer. The result of this script is a json file with the contributors collect for each repository if this data was available.

## Mine commits

Mine commits is the main step where the features are being extracted from commits. For each repository:
- The repository is downloaded to the specified path
- The commits are imported and put into an array
- File and commit level features are being calculated by the CommitMiner. They are cached in the specified location.
- The final features are calculated (copied or aggregated) by the FeatureCalculator. 
- The repository is remove to save space

By modifying the AbstractMiner.features field one can limit the features that are calculated, greatly reducing the calculation time if specific features are not required.

## Group and annotate

During the mining process the label is not propagated -- the information whenever the commit comes from the set of security-related commits or from background is not saved. As such the mined data is one more time crossed with the input set of commits to label the datapoints.
 
The final stage of the pipeline is to concatinate all dataframes of each repository into one single dataset. We also seperate out the commits that are changing the files in specific programming language (based on extensions) to create the per-ecosystem subsets.
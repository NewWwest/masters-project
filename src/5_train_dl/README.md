# Training Deep Learning

This directory contains the start scripts to execute the deep learning experiment.

### Fold Data

The fold data script should be used on the output of `3_extract_sample_from_zips` to split the data into n chunks, from which the second to last chunk will be used for validation while the last for test.

### Train

The train script contains the entry point for the trraining and evaluation pipeline. It contains two functions `do_fold_finetuning` and `do_folds_aggregators` which can be called to run a given stage of the experiment. They use the functions defined in `training_utils` file in the `dl` directory, where more information can be found.
do_fold_finetuning()
lstm_folds_results, conv_folds_results, mean_folds_results = do_folds_aggregators()

### Calculate AUC and Calculate RC5

These two files use the classification resutls from the VulCurator models to calculated the more advanced metrics of area under the precision-recall curve and recall at 5% of reviewed lines.
# Deep Learning

This directories contains general classes used in deep learning experiments.

## Models

Models contains the PyTorch definitions of the used models in fine-tuning, training and evaluation.
- `BertAndLinear` is the model used to fine-tune GraphCodeBert model. It is the transformer with a linerab layer attached to it.
- `FrozenBertAndLinear` is the same model as `BertAndLinear`, but the weight in the transformer are frozen. Can be used for experimentation or for evaluation.
- `ConvAggregator` is the implementation of the Convolution Aggregator described more in the thesis.
- `MeanAggregator` is the implementation of the Mean Aggregator described more in the thesis, which mimics the architecture of VulFixMiner.
- `LstmAggregator` is the implementation of the Lstm Aggregator described more in the thesis, which is simialar in the implementation as the model used by Commit2Vec.

The aggregators are  implemented to handle a sequence of maximum 100 embeddings - this is also the number of samples we maximally extract from commits during mining.

## Datasets

The Datasets directory contains implementation of the Dataset interface that can be used to loader the mined data from the disk to be used in traing and or evaluation.

### BaseRawDataset & CommitLevelRawDataset & SampleLevelRawDataset

These classes directly operate on the files in the analyzed directory. Positive data is kept separate from the background to easy with the resampling. The two concreate implementations of the base class (CommitLevelRawDataset vs SampleLevelRawDataset) differ by the way they are importing samples -- commit level dataset contains them as a sequence of sequences of inputs, while the sample level dataset ccontains them in a flattened form as a sequence of inputs. Additionally function in the `load` file can simplify the loading and splitting of the data.

### Sampling

The `OverSampledDataset` and `UnderSampledDataset` provide a way to sample the raw datasets to a specific ratio of samples. Since in the final training pipeline the splitting is done on a file-level basis these classes aren't used in the end (?).

### Supporting

Supporting datasets are the `CsvDataset` (used to save intermediate datasets) and the `DatasetRandom` used to test the models in the beginning.
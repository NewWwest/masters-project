# Training Deep Learning


## Scripts

The scripts were the first iteration of the deep learning experiment. Each stage (as depicked below) can be examined in more detail and experimented on

### Fine-Tune
The fine-tuning step involves training the transformed to produce the best possible mebedding of the provided samples for the specific task. To do that we train a model build from the GraphCodeBERT transformer with a linear layer attached to it. The model is train for cross entropy loss and after trainng the linear layer is discarded.

### Embed tokens
In the embedding process the fine-tuned transformer is used on the mined commit samples to convert them to embeddings. 

### Train Aggregator
In this stage the produced embeddings are fed into the aggregator models which combine them into one single result.

## Notebooks

### CrossVal

The CrossVal notebook allows for cross validation training and evaluation of the models. As the process of processing the tokenized imputs into their embeddings is time consuming, 

### Test transformer hyper-parameters

This notebook contains the code to evaluate to embedding transformer different configurations of under- and over-sampling and differen tlearning rate which were identified as the main contributors to it's performance (apart from the used data). The test should take place on limitted datasets to evaluate one sampling method 32 train adn test cycles need to be calculated.


## Results

The `results` folder contains the outputs of the search for the optimal hyper-parameters setting using the `Test transformer hyper-parameters` notebook. The individual outputs are contained in the text files named after the used sampling method while the .ods and .xls file contain the contatinated results in tables.
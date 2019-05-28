# ThermoNet
This is a TensorFlow implementation accompanying our paper. This codebase is based on Shreshthgandhi's Tensorflow implementation of the cdeepbind model. The framework of model training and testing have beed adopted with minor changes. Other code files have been modified and re-structured with changes specific to our model. And the folder, `scripts/RNAsubopt` is a copy of RNAsubopt from the ViennaRNA  project.

## Prepare the Training Data
We used two datasets to evaluate our model.
The following datsets were used for training our models.
* [RNAcompete]
* [CLIP-seq](https://github.com/mstrazar/iONMF)

You can download the datasets from the corresponding website. 

After that， you should prepare the data used in the training code according to the steps below

1.Clean the dataset format

Use the `python scripts/0_get_pure_seq_and_label.py` and `python scripts/1_combine_train_test.py`

2.Sample 100 possible secondary strucutures

Use the `python scripts/2_generate_top100.py`

3.Generate sequence embedding id

Use the `python scripts/3_generate_embedding.py`

## Run the Training Script

Use the `run.sh` script to train a model. 
The following variables have to be specified.

```
* DATA_DIR      # Path to TFRecord files
* RESULTS_HOME  # Directory to store results
* CFG           # Name of model configuration 
* MDL_CFGS      # Path to model configuration files
* GLOVE_PATH    # Path to GloVe dictionary and embeddings
```

Example configuration files are provided in the model\_configs folder. During training, model files will be stored under a directory named `$RESULTS_HOME/$CFG`.

### Training using pre-trained word embeddings

The implementation supports using fixed pre-trained GloVe word embeddings.
The code expects a numpy array file consisting of the GloVe word embeddings named `glove.840B.300d.npy` in the `$GLOVE_PATH` folder.

## Evaluating a Model

### Expanding the Vocabulary

Once the model is trained, the vocabulary used for training can be optionally expanded to a larger vocabulary using the technique proposed by the SkipThought paper. 
The `voc_exp.sh` script can be used to perform expansion. 
Since Word2Vec embeddings are used for expansion, you will have to download the Word2Vec model. 
You will also need the gensim library to run the script.

### Evaluation on downstream tasks

Use the `eval.sh` script for evaluation. The following variables need to be set.

```
* SKIPTHOUGHTS  # Path to SkipThoughts implementation
* DATA          # Data directory for downstream tasks
* TASK          # Name of the task
* MDLS_PATH     # Path to model files
* MDL_CFGS      # Path to model configuration files
* CFG           # Name of model configuration 
* GLOVE_PATH    # Path to GloVe dictionary and embeddings
```

Evaluation scripts for the downstream tasks from the authors of the SkipThought model are used. These scripts train a linear layer on top of the sentence embeddings for each task. 
You will need to clone or download the [skip-thoughts GitHub repository](https://github.com/ryankiros/skip-thoughts) by [ryankiros](https://github.com/ryankiros).
Set the `DATA` variable to the directory containing data for the downstream tasks. 
See the above repository for further details regarding downloading and setting up the data.

To evaluate the pre-trained models, set the directory variables appropriately.
Set `MDLS_PATH` to the directory of downloaded models.
Set the configuration variable `CFG` to one of 
* `MC-BC` (Multi-channel BookCorpus model) or 
* `MC-UMBC` (Multi-channel BookCorpus + UMBC model)

Set the `TASK` variable to the task of interest.

## Reference

If you found our code useful, please cite us [1](https://arxiv.org/pdf/1803.02893.pdf).

```
@inproceedings{
logeswaran2018an,
  title={An efficient framework for learning sentence representations},
  author={Lajanugen Logeswaran and Honglak Lee},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJvJXZb0W},
}
```

Contact: [llajan@umich.edu](mailto:llajan@umich.edu)

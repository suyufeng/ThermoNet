# ThermoNet
This is a TensorFlow implementation accompanying our paper. This codebase is based on Shreshthgandhi's Tensorflow implementation of the cdeepbind model. The framework of model training and testing have beed adopted with minor changes. Other code files have been modified and re-structured with changes specific to our model. And the folder, `scripts/RNAsubopt` is a copy of RNAsubopt from the ViennaRNA  project.

## Prepare the Training Data
We used two datasets to evaluate our model.
The following datsets were used for training our models.
* [RNAcompete]
* [CLIP-seq](https://github.com/mstrazar/iONMF)

You can download the datasets from the corresponding website. 
After that, you should prepare the data used in the training code according to the steps below. We use the CLIP-seq dataset as an example.

### Clean the dataset format
Use the `python scripts/0_get_pure_seq_and_label.py` and `python scripts/1_combine_train_test.py`

### Sample 100 possible secondary structures
Use the `python scripts/2_generate_top100.py` and `python scripts/4_generate_structure_message.py`

### Generate embedding id
Use the `python scripts/3_generate_embedding.py`

### Extract top 5 secondary structures 
Use the `python scripts/5_generate_top5.py`

You will get six kinds of data, which are ".pure.seq", ".label", "\_combine.map.top100.npy", "\_combine.map.top5.npz", "n_gram/\_mer.npy" and "\_combine.top5.prob.npy". You need to link the addresses of the six files in the code to the corresponding addresses.

## Run the Main Script
Use the `python invitro/bin/train_model_update.py` to train the model in *in vitro* dataset and the `python invivo/bin/train_model_update.py` to train the model in *in vivo* dataset. After the code is completed, we will save the best hyperparameter and results. You can change the saving address in `configs/cnn_config.yml`. 

BTW, The ThermoNet implementation is available at `invitro/bin/deepbind_model/utils_update.py`. If you just want to see how to implemente ThermoNet instead of runing the code, you can check the ThermoNet class directly.


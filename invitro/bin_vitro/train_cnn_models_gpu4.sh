#!/usr/bin/env bash
nohup python train_model_update.py --protein RNCMPT00068 RNCMPT00069 RNCMPT00070 RNCMPT00071 RNCMPT00072 RNCMPT00073 RNCMPT00074 --configuration ../configs/cnn_config.yml --gpus 0&
nohup python train_model_update.py --protein RNCMPT00076 RNCMPT00077 RNCMPT00078 RNCMPT00079 RNCMPT00080 RNCMPT00081 RNCMPT00082 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein RNCMPT00084 RNCMPT00085 RNCMPT00086 RNCMPT00087 RNCMPT00088 RNCMPT00089 RNCMPT00090 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein RNCMPT00093 RNCMPT00094 RNCMPT00095 RNCMPT00096 RNCMPT00097 RNCMPT00099 RNCMPT00205 --configuration ../configs/cnn_config.yml --gpus 3&
nohup python train_model_update.py --protein RNCMPT00102 RNCMPT00103 RNCMPT00104 RNCMPT00100 RNCMPT00106 RNCMPT00107 RNCMPT00116 --configuration ../configs/cnn_config.yml --gpus 0&
nohup python train_model_update.py --protein RNCMPT00112 RNCMPT00108 RNCMPT00109 RNCMPT00110 RNCMPT00111 RNCMPT00113 RNCMPT00114 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein RNCMPT00121 RNCMPT00118 RNCMPT00119 RNCMPT00120 RNCMPT00122 RNCMPT00123 RNCMPT00124 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein RNCMPT00127 RNCMPT00129 RNCMPT00131 RNCMPT00132 RNCMPT00133 RNCMPT00134 RNCMPT00136 --configuration ../configs/cnn_config.yml --gpus 3&

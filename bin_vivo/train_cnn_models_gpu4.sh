#!/usr/bin/env bash
nohup python train_model_update.py --protein invivo_1 invivo_2 invivo_3 invivo_4 --configuration ../configs/cnn_config_invivo.yml --gpus 0&
nohup python train_model_update.py --protein invivo_5 invivo_6 invivo_7 invivo_8 --configuration ../configs/cnn_config_invivo.yml --gpus 0&
nohup python train_model_update.py --protein invivo_9 invivo_10 invivo_11 invivo_12 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein invivo_13 invivo_14 invivo_15 invivo_16 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein invivo_17 invivo_18 invivo_19 invivo_20 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein invivo_21 invivo_22 invivo_23 invivo_24 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein invivo_25 invivo_26 invivo_27 invivo_29 --configuration ../configs/cnn_config.yml --gpus 3&
nohup python train_model_update.py --protein invivo_30 invivo_31 invivo_28 --configuration ../configs/cnn_config.yml --gpus 3&

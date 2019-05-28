#!/usr/bin/env bash
nohup python train_model_update.py --protein RNCMPT00137 RNCMPT00138 RNCMPT00139 RNCMPT00140 RNCMPT00141 RNCMPT00142 RNCMPT00143 RNCMPT00144 --configuration ../configs/cnn_config.yml --gpus 0&
nohup python train_model_update.py --protein RNCMPT00145 RNCMPT00146 RNCMPT00147 RNCMPT00148 RNCMPT00149 RNCMPT00150 RNCMPT00151 RNCMPT00152 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein RNCMPT00153 RNCMPT00154 RNCMPT00155 RNCMPT00156 RNCMPT00157 RNCMPT00158 RNCMPT00159 RNCMPT00160 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein RNCMPT00161 RNCMPT00162 RNCMPT00163 RNCMPT00164 RNCMPT00165 RNCMPT00166 RNCMPT00167 RNCMPT00168 --configuration ../configs/cnn_config.yml --gpus 3&
nohup python train_model_update.py --protein RNCMPT00172 RNCMPT00169 RNCMPT00170 RNCMPT00171 RNCMPT00173 RNCMPT00174 RNCMPT00175 RNCMPT00176 --configuration ../configs/cnn_config.yml --gpus 0&
nohup python train_model_update.py --protein RNCMPT00177 RNCMPT00178 RNCMPT00179 RNCMPT00180 RNCMPT00181 RNCMPT00182 RNCMPT00183 RNCMPT00184 --configuration ../configs/cnn_config.yml --gpus 1&
nohup python train_model_update.py --protein RNCMPT00185 RNCMPT00186 RNCMPT00187 RNCMPT00197 RNCMPT00199 RNCMPT00200 RNCMPT00202 RNCMPT00203 --configuration ../configs/cnn_config.yml --gpus 2&
nohup python train_model_update.py --protein RNCMPT00206 RNCMPT00209 RNCMPT00212 RNCMPT00215 RNCMPT00216 RNCMPT00217 RNCMPT00218 RNCMPT00219 --configuration ../configs/cnn_config.yml --gpus 3&

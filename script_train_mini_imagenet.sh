#!/usr/bin/env bash

python train_mini_imagenet.py \
--arch default_convnet \
--epochs 200 \
--print-freq 20 \
--optimizer adam \
--step_size 20 \
--gamma 0.5 \
--alpha 0.5 \
--lr 0.001 \
-j 24 \
--n_query_train 15 \
--n_query_val 15 \
--n_support 1 \
--n_way_train 30 \
--n_way_val 5 \
--n_episodes_train 100 \
--n_episodes_val 400 \
--model_name mini_imagenet_1_shot_30_way_alpha_0.5

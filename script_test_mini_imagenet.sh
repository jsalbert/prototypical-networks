#!/usr/bin/env bash

python test_mini_imagenet.py \
--gpu 0 \
--arch default_convnet \
--workers 24 \
--n_episodes 2000 \
--n_way 5 \
--n_support 1 \
--n_query 30 \
--checkpoint 'models_trained/mini_imagenet_1_shot_30_way/model_best_acc.pth.tar' \
--evaluation_name 'mini_imagenet_1_shot_30_way'
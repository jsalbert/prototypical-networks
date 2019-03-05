#!/usr/bin/env bash

python test_mini_imagenet.py \
--gpu 0 \
--arch default_convnet \
--workers 24 \
--n_episodes 2000 \
--n_way 5 \
--n_support 5 \
--n_query 30 \
--checkpoint 'models_trained/mini_imagenet_5_shot_20_way/model_best_acc.pth.tar' \
--evaluation_name 'minimagenet_5_shot_20_way' \
--results_name 'testing'
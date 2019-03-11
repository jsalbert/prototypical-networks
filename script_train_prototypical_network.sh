#!/usr/bin/env bash

python train.py \
--arch resnet50 \
--epochs 200 \
--pretrained \
--weight-decay 0.00001 \
--print-freq 20 \
--lr 0.001 \
--optimizer 'adam' \
--gamma 0.5 \
--alpha 0.5 \
--step_size 30 \
--workers 24 \
--subtract_mean 156.2336961908687 122.03200584422879 109.9825961313363 \
--subtract_std 46.39668432 42.3512562 41.54967605 \
--image_size 224 \
--n_query_train 15 \
--n_query_val 5 \
--n_support 5 \
--n_way_train 20 \
--n_way_val 20 \
--n_episodes_train 200 \
--n_episodes_val 400 \
--train_dir /data/datasets/ \
--val_dir /data/datasets/ \
--model_name ResNet50
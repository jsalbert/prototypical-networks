#!/usr/bin/env bash

python test.py \
--gpu 0 \
--arch resnet50 \
--workers 24 \
--batch_size 64 \
--subtract_mean 156.2336961908687 122.03200584422879 109.9825961313363 \
--subtract_std 46.39668432 42.3512562 41.54967605 \
--image_size 224 \
--train_dir /data/datasets/ \
--test_dir /data/datasets/ \
--checkpoint 'models_trained/ResNet50/model_best_acc.pth.tar' \
--evaluation_name 'Resnet50_evaluation' \
--results_name 'testing' \
--save_prototypes
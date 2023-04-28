#!/usr/bin/env bash

# TODO: can try training with and without augmentation with synthetic data

# train base classifier
# NOTE: without augmentation with just five image training, acc is 16.49 in 50 epochs, and 16.22 in 100 epochs with augmentation (just tencrop)
#python train_clf.py \
#    --train_dir 'train' \
#    --gpu 0 \
#    --epochs 100 \
#    --lr 0.01 \
#    --overwrite

#python train_clf.py \
#    --train_dir 'train' \
#    --gpu 0 \
#    --epochs 100 \
#    --lr 0.01 \
#    --augment \
#    --overwrite

# train base classifier v2
#python train_clf.py \
#    --train_dir 'train2' \
#    --val_dir 'val2' \
#    --gpu 0 \
#    --epochs 100 \
#    --lr 0.01 \
#    --overwrite

# this baseline will contain the same number of training samples as the synthetic data augmented approaches
#python train_clf.py \
#    --train_dir 'train2' \
#    --val_dir 'val2' \
#    --gpu 0 \
#    --bs 64 \
#    --epochs 100 \
#    --lr 0.01 \
#    --augment \
#    --overwrite

# train with synthetic (pti) data
#python train_clf.py \
#    --train_dir 'pti_train' \
#    --val_dir 'val' \
#    --gpu 0 \
#    --epochs 100 \
#    --lr 0.01 \
#    --overwrite

python train_clf.py \
    --train_dir 'pti_train' \
    --val_dir 'val' \
    --augment \
    --bs 64 \
    --gpu 0 \
    --epochs 100 \
    --lr 0.01 \
    --overwrite

# train with synthetic (lora) data
#python train_clf.py \
#    --train_dir 'lora_train' \
#    --val_dir 'val' \
#    --gpu 1 \
#    --epochs 100 \
#    --lr 0.01 \
#    --overwrite

#python train_clf.py \
#    --train_dir 'lora_train' \
#    --val_dir 'val' \
#    --augment \
#    --bs 64 \
#    --gpu 1 \
#    --epochs 100 \
#    --lr 0.01 \
#    --overwrite

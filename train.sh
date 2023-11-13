#!/usr/bin/bash
export HTTPS_PROXY=http://localhost:7890/
export HTTP_PROXY=http://localhost:7890/
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

# 1 epoch of >20 samples
accelerate launch train.py --batch_size 128  --save_interval 1 --checkpoint /Projects/Github/paper_project/EDGE/checkpoint.pt --start_epoch 0 --epochs 1 --feature_type jukebox --learning_rate 0.00001 --use_beats_anno --freeze_layers --use_music_beat_feat

# 1 epoch of 20 samples, give best PFC results
accelerate launch train.py --batch_size 84  --save_interval 1 --checkpoint /Projects/Github/paper_project/EDGE/checkpoint.pt --start_epoch 0 --epochs 1 --feature_type jukebox --learning_rate 0.00001 --use_beats_anno --freeze_layers --use_music_beat_feat
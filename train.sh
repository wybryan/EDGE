export HTTPS_PROXY=http://localhost:7890/
export HTTP_PROXY=http://localhost:7890/
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch train.py --batch_size 128  --checkpoint  /Projects/Github/paper_project/EDGE/runs/train/exp3/weights/train-300.pt --save_interval 50 --start_epoch 300 --epochs 2000 --feature_type jukebox --learning_rate 0.0002
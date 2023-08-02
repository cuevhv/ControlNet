export PYTHONBUFFERED=1
export PATH=$PATH
export TRANSFORMERS_OFFLINE=1
/home/hcuevas/miniconda3/envs/control2/bin/python train_apple.py --model_cfg_yaml $1 --model_checkpoint $2 --dataset_prompts_json $3 --batch_size 16 --gpus 8 --workers 16
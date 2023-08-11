export PYTHONBUFFERED=1
export PATH=$PATH
export TRANSFORMERS_OFFLINE=1
/home/hcuevas/miniconda3/envs/control_pl/bin/python train_bedlam.py --control_type segment_human_and_clothes --model_cfg_yaml $1 --model_checkpoint $2 --train_dataset_prompts_json $3 --val_dataset_prompts_json $4 --batch_size 16 --gpus 4 --workers 16
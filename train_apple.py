""" TRANSFORMERS_OFFLINE=1 python train_apple.py --model_cfg_yaml models/cldm_v21_evermotion_seg.yaml \
    --model_checkpoint models/control_sd21_ini_evermotion_seg.ckpt --dataset_prompts_json dataset/evermotion_dataset/prompt.json \
    --batch_size 1 --gpus 1 --workers 0
"""
from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataloaders.dataset_apple import EvermotionDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os


def get_float_precision(is_half_precision: bool):
    precision = 32
    if torch.cuda.get_device_properties(torch.cuda.device(0)).major < 7:
        print("GPU does not support half precision (16fp), using single precision (32fp)")
        precision = 32
    elif is_half_precision == True:
        precision = 16
        print("Using half precision (16fp)")
    else:
        print("Using single precision (32fp)")
    return precision


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_cfg_yaml", type=str, help="Path to control model configuration yaml file")
    args.add_argument("--model_checkpoint", type=str, help="Path to control model checkpoint or SD(pretrained)+controlnet(new) checkpoint")
    args.add_argument("--dataset_prompts_json", type=str, help="Path to a json file with source, control and prompt file locations")
    args.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args.add_argument("--gpus", type=int, default=1, help="Number of GPUs to be used")
    args.add_argument("--workers", type=int, default=0, help="Batch size")
    args.add_argument("--half_precision", action="store_true", help="Use half precision")
    
    return args.parse_args()


def main():
    args = parse_args()

    # Configs
    resume_path = args.model_checkpoint 
    batch_size = args.batch_size
    num_workers = args.workers
    logger_freq = 2000
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    checkpoints_dir = 'checkpoints'

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.model_cfg_yaml).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # dataset
    dataset = EvermotionDataset(args.dataset_prompts_json)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    
    logger = ImageLogger(batch_frequency=logger_freq)
    os.mkdirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=logger_freq,
                                          dirpath=checkpoints_dir,
                                            filename='evermotion-{step:02d}-{epoch:02d}-{val_loss:.2f}')

    # Train!
    trainer = pl.Trainer(gpus=args.gpus, precision=get_float_precision(args.half_precision), callbacks=[logger, checkpoint_callback])
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
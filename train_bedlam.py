""" TRANSFORMERS_OFFLINE=1 python train_bedlam.py --model_cfg_yaml models/cldm_v15_bedlam50_seg_body_clothes.yaml \
    --model_checkpoint models/control_sd15_ini_bedlam50_seg_body_clothes.ckpt \
    --train_dataset_prompts_json dataset/20230804_1_3000_hdri/prompt_train.json \
    --val_dataset_prompts_json dataset/20230804_1_3000_hdri/prompt_val.json \
    --batch_size 1 --gpus 1 --workers 0 --control_type segment_human_and_clothes
"""
from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataloaders.dataset_bedlam import BedlamSimpleDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from dataloaders.pl_dataloader_bedlam import BedlamDataModule
import yaml
#from cfgnode import CfgNode
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
    args.add_argument("--train_dataset_prompts_json", type=str, help="Path to a json file with source, control and prompt file locations")
    args.add_argument("--val_dataset_prompts_json", type=str, help="Path to a json file with source, control and prompt file locations")
    args.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args.add_argument("--gpus", type=int, default=1, help="Number of GPUs to be used")
    args.add_argument("--workers", type=int, default=0, help="Batch size")
    args.add_argument("--half_precision", action="store_true", help="Use half precision")
    args.add_argument("--control_type", type=str, default="segmentation")
    # args.add_argument("--script_config_yaml", type=str, help="Path to yaml file with the configuration for the script")

    return args.parse_args()


def main():
    args = parse_args()

    # TODO: add in the future when we wrap up Read config file.
    # cfg = None
    # with open(args.script_config_yaml, "r") as f:
    #     cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    #     cfg = CfgNode(cfg_dict)

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

    use_pl_datamoule = True
    # dataset
    if use_pl_datamoule:
        data_module = BedlamDataModule(train_data_json=args.train_dataset_prompts_json, val_data_json=args.val_dataset_prompts_json,
                                    control_type=[args.control_type], train_dataloader_conf={'batch_size': batch_size,
                                                                                   'num_workers': num_workers,
                                                                                   'shuffle': True})
    else:
        train_dataset = BedlamSimpleDataset(args.train_dataset_prompts_json, condition_type=[args.control_type])
        train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)

        val_dataset = BedlamSimpleDataset(args.val_dataset_prompts_json, condition_type=[args.control_type])
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

        print("Training images: ", len(train_dataset))


    logger = ImageLogger(batch_frequency=logger_freq, control_type=args.control_type, max_images=8)
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=logger_freq,
                                          dirpath=checkpoints_dir,
                                            filename='evermotion-{step:02d}-{epoch:02d}-{val_loss:.2f}')

    # Train!
    if pl.__version__ == "2.0.6":
        trainer = pl.Trainer(accelerator="gpu", devices=args.gpus, strategy="ddp_find_unused_parameters_true", precision=get_float_precision(args.half_precision),
                    callbacks=[logger, checkpoint_callback], accumulate_grad_batches=2)

    else:
        trainer = pl.Trainer(gpus=args.gpus, precision=get_float_precision(args.half_precision),
                            callbacks=[logger, checkpoint_callback], accumulate_grad_batches=2)

    if use_pl_datamoule:
        trainer.fit(model, data_module)
    else:
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()

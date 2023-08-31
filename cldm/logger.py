import os

import numpy as np
import torch
import torchvision
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
if pl.__version__ == "2.0.6":
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
else:
    from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_only
from distinctipy import distinctipy

COLORS = np.array(distinctipy.get_colors(100))
COLORS[0] = [0, 0, 0]

class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, control_type="segmentation"):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {"N": self.max_images}
        self.log_first_step = log_first_step
        self.control_type = control_type


    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, condition_names, condition_sizes):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            image = images[k]
            if k == "control": #images[k].shape[1] > 3:
                image = images[k]
                for cond_idx, condition_name in enumerate(condition_names):
                    condition_size = condition_sizes[cond_idx]
                    if cond_idx == 0:
                        start_idx = 0
                        end_idx = 0
                    else:
                        start_idx = end_idx
                    end_idx = end_idx + condition_size
                    individual_condition_img = image[:, start_idx:end_idx]
                    if condition_name in ["segment_human_and_clothes", "ldmks2d"]:
                        n_clases = individual_condition_img.shape[1]
                        color_multiplier = 1. / (n_clases - 1)
                        condition_indices = individual_condition_img.argmax(1, keepdim=False).numpy()
                        individual_condition_img = torch.from_numpy(COLORS[condition_indices]).type(image.dtype).permute(0, 3, 1, 2)
                    # breakpoint()
                    grid = torchvision.utils.make_grid(individual_condition_img, nrow=4)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k+"_"+condition_name, global_step, current_epoch, batch_idx)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)

            else:
                grid = torchvision.utils.make_grid(image, nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            condition_names = batch['condition_names'][0].split(',')
            condition_sizes = batch['condition_sizes'][0].detach().cpu().numpy().tolist()

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, condition_names, condition_sizes)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")

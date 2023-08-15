from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import matplotlib.pyplot as plt


apply_uniformer = UniformerDetector()

model = create_model('./models/cldm_v21_bedlam50_seg_body_clothes.yaml').cpu()
state_dict_fn = "/home/hcuevas/Documents/work/gen_bedlam/external_repos/ControlNet/checkpoints/bedlam-step-8000-epoch-19-val_loss-0.00.ckpt"
model.load_state_dict(load_state_dict(state_dict_fn, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(detected_map, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        H, W, C = detected_map.shape

        control = torch.from_numpy(detected_map.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results




condition_body_filename = "dataset/20230804_1_3000_hdri/exr_layers/masks_untar/20230804_1_3000_hdri_masks.9/20230804_1_3000_hdri/exr_layers/masks/seq_002704/seq_002704_0000_00_body.png"
condition_clothe_filename = "dataset/20230804_1_3000_hdri/exr_layers/masks_untar/20230804_1_3000_hdri_masks.9/20230804_1_3000_hdri/exr_layers/masks/seq_002704/seq_002704_0000_00_clothing.png"
condition_masks_body = cv2.imread(condition_body_filename, cv2.IMREAD_GRAYSCALE)
condition_masks_clothe = cv2.imread(condition_clothe_filename, cv2.IMREAD_GRAYSCALE)

condition_masks_body = condition_masks_body/255.
condition_masks_clothe = condition_masks_clothe/255.

condition_masks = condition_masks_body + condition_masks_clothe

condition_env = 1 - condition_masks
condition_masks = np.stack([condition_env, condition_masks_body, condition_masks_clothe], axis=-1)
prompt = 'a person in a scene'
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 50
guess_mode = False  # TODO: check this
strength = 1.
scale = 9.
seed = 2147483647
eta = 0.0




out = process(condition_masks, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
print(out)
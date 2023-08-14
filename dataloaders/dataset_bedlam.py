import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
from dataloaders import custom_transforms
import os

class BedlamSimpleDataset(Dataset):
    def __init__(self, imgs_prompts_fn: str, condition_type = ["segment_human_and_clothes"]):
        self.data = []

        self.condition_type = condition_type
        with open(imgs_prompts_fn, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.transforms = custom_transforms.Compose([
            # TODO: remove fit_to_new_size
            custom_transforms.RandomResizeCrop(512, scale=(0.9, 1), fit_to_new_size=True),
            custom_transforms.RandomHorizontalFlip(prob=0.5),
            custom_transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.data)


    def get_segment_human_masks(self, target_fn: str, single_channel: bool = False):
        folder_1 = "hdri_masks"
        folder_2 = os.path.join("exr_layers", "masks")
        # /home/hcuevas/Documents/work/gen_bedlam/datasets/20230804_1_3000_hdri/exr_layers/masks_untar/20230804_1_3000_hdri_masks.0/20230804_1_3000_hdri/exr_layers/masks/seq_000000/seq_000000_0000_00_env.png
        condition_fn = target_fn.replace("hdri_png", folder_1).replace("png", folder_2)
        condition_fn = condition_fn.rsplit(".", 1)
        condition_body_filename = condition_fn[0]+"_00_body.png"
        condition_clothe_filename = condition_fn[0]+"_00_clothing.png"

        condition_masks_body = cv2.imread(condition_body_filename, cv2.IMREAD_GRAYSCALE)
        condition_masks_clothe = cv2.imread(condition_clothe_filename, cv2.IMREAD_GRAYSCALE)

        # Some images do not have body or clothes masks so we create a black mask for them
        if condition_masks_body is None:
            condition_masks_body = np.zeros((512, 512), dtype=np.uint8)
        if condition_masks_clothe is None:
            condition_masks_clothe = np.zeros((512, 512), dtype=np.uint8)

        condition_masks_body = condition_masks_body/255.
        condition_masks_clothe = condition_masks_clothe/255.

        condition_masks = condition_masks_body + condition_masks_clothe

        if single_channel is False:
            condition_env = 1 - condition_masks
            condition_masks = np.stack([condition_env, condition_masks_body, condition_masks_clothe], axis=-1)

        return condition_masks, condition_fn


    def get_depth(self, target_filename: str, inverse_depth: bool = True, to_cm: bool = True):
        condition_filename = target_filename.replace("final_preview", "geometry_hdf5")
        condition_filename = condition_filename.replace("tonemap.jpg", "depth_meters.hdf5")

        with h5py.File(condition_filename, "r") as hdf5_file:
            condition_depth = np.asarray(hdf5_file["dataset"], dtype=np.float32)

        if to_cm:
           condition_depth = condition_depth*100

        condition_depth = 1./condition_depth
        condition_depth -= np.nanmin(condition_depth)
        condition_depth /= np.nanmax(condition_depth)
        condition_depth = np.nan_to_num(condition_depth, nan=0)

        return condition_depth


    def get_condition(self, target_filename: str):
        if "segment_human_env" in self.condition_type:
            # Returns single mask with human vs envirnoment
            condition, condition_fn = self.get_segment_human_masks(target_filename, single_channel=True)
            condition = np.repeat(condition[:,:,None], 3, axis=-1)

        elif "segment_human_and_clothes" in self.condition_type:
            # Returns three channel masks: human, clothes, environment
            condition, condition_fn = self.get_segment_human_masks(target_filename)

        elif "canny" in self.condition_type:
            raise NotImplementedError

        elif "depth" in self.condition_type:
            raise NotImplementedError
        else:
            raise NotImplementedError

        return condition, condition_fn


    def __getitem__(self, idx: int):
        item = self.data[idx]

        " prompt will be only scene 20% of the time"
        prompt = item['prompt'] if np.random.random() < 0.8 else "person in a scene"

        target_filename = item['target']
        target_rgb = cv2.imread(target_filename)
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        img_conditions, conditions_fn = self.get_condition(target_filename)
        img_conditions = img_conditions.astype(np.float32)

        # Normalize target_rgb images to [-1, 1].
        target_rgb = (2*target_rgb.astype(np.float32) / 255) - 1.0
        target_rgb, img_conditions = self.transforms(target_rgb, img_conditions)

        return dict(jpg=target_rgb, txt=prompt, hint=img_conditions)

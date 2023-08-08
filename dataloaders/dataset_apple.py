import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
from dataloaders import custom_transforms


class EvermotionDataset(Dataset):
    """ NYC40 labels -1 background and 40 classes from 1 to 40 inclusive
        https://github.com/apple/ml-hypersim/issues/12#issuecomment-759720323
    """
    def __init__(self, imgs_prompts_fn: str, condition_type = ["segmentation"]):
        self.data = []

        self.n_labels = 40
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


    def get_seg_color(self, target_filename: str):
        condition_filename = target_filename.replace("final_preview", "geometry_hdf5")
        condition_filename = condition_filename.replace("tonemap.jpg", "semantic.hdf5")

        with h5py.File(condition_filename, "r") as hdf5_file:
            condition_seg = np.asarray(hdf5_file["dataset"], dtype=np.float32)

        condition_seg_labels = np.zeros((condition_seg.shape[0], condition_seg.shape[1], self.n_labels))
        unique_labels = np.unique(condition_seg).astype(np.int32)

        for label in unique_labels:
            if label == -1:
                continue
            condition_seg_labels[:, :, label-1] = (condition_seg == label)
        return condition_seg_labels


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
        if "segmentation" in self.condition_type:
            condition = self.get_seg_color(target_filename)
        elif "canny" in self.condition_type:
            condition = cv2.imread(target_filename)
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2GRAY)
            condition = cv2.Canny(condition, threshold1=100, threshold2=200)
            condition = condition/255.0
            condition = np.repeat(condition[:,:,None], 3, axis=-1)
        elif "depth" in self.condition_type:
            condition = self.get_depth(target_filename, inverse_depth=True, to_cm=True)
            condition = np.repeat(condition[:,:,None], 3, axis=-1)
        else:
            print(self.condition_type)
            raise NotImplementedError
        return condition


    def __getitem__(self, idx: int):
        item = self.data[idx]

        " prompt will be only scene 20% of the time"
        prompt = item['prompt'] if np.random.random() < 0.8 else "scene"

        target_filename = item['target']
        target_rgb = cv2.imread(target_filename)
        target_rgb = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2RGB)

        img_conditions = self.get_condition(target_filename).astype(np.float32)

        # Normalize target_rgb images to [-1, 1].
        target_rgb = (2*target_rgb.astype(np.float32) / 255) - 1.0
        target_rgb, img_conditions = self.transforms(target_rgb, img_conditions)

        return dict(jpg=target_rgb, txt=prompt, hint=img_conditions)

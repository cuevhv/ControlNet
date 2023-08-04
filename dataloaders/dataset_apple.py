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
        condition_filename = condition_filename.replace("color.jpg", "semantic.hdf5")

        with h5py.File(condition_filename, "r") as hdf5_file:
            condition_seg = np.asarray(hdf5_file["dataset"], dtype=np.float32)

        condition_seg_labels = np.zeros((condition_seg.shape[0], condition_seg.shape[1], self.n_labels))
        unique_labels = np.unique(condition_seg).astype(np.int32)

        for label in unique_labels:
            if label == -1:
                continue
            condition_seg_labels[:, :, label-1] = (condition_seg == label)
        return condition_seg_labels


    def get_condition(self, target_filename: str):
        if "segmentation" in self.condition_type:
            condition_seg = self.get_seg_color(target_filename)
        else:
            raise NotImplementedError
        return condition_seg


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

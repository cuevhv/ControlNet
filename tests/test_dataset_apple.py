import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append(".")

from dataloaders.dataset_apple import EvermotionDataset

if __name__ == "__main__":
    condition_type = "canny"
    dataset = EvermotionDataset("dataset/evermotion_dataset/prompt.json", ["canny"])
    dataloader = DataLoader(dataset, num_workers=2, batch_size=4, shuffle=True)
    print("dataset images: ", len(dataset))
    print("dataloader length: ", len(dataloader))
    print("batch size: ", dataloader.batch_size)
    print("dataloader length*batch_size: ", dataloader.batch_size*len(dataloader))
    for data in dataloader:
        for i in range(len(data["txt"])):
            print(data.keys(), data["jpg"].shape, data["hint"].shape)
            rgb_img = (data["jpg"][i].numpy() + 1) / 2 
            print(np.max(rgb_img), np.min(rgb_img))
            plt.subplot(1,2,1), plt.imshow(rgb_img)
            condition = data["hint"][i].numpy()
            if condition_type == "segmentation":
                condition = condition.argmax(-1)
            plt.subplot(1,2,2), plt.imshow(condition)
            plt.title(data["txt"][i])
            plt.show()

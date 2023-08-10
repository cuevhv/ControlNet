import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.append(".")

from dataloaders.dataset_bedlam import BedlamSimpleDataset

if __name__ == "__main__":
    condition_type = "segment_human_and_clothes"
    dataset = BedlamSimpleDataset("dataset/20230804_1_3000_hdri/caption.json", [condition_type])
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    print("dataset images: ", len(dataset))
    print("dataloader length: ", len(dataloader))
    print("batch size: ", dataloader.batch_size)
    print("dataloader length*batch_size: ", dataloader.batch_size*len(dataloader))
    for j, data in enumerate(dataloader):
        for i in range(len(data["txt"])):
            print(data.keys(), data["jpg"].shape, data["hint"].shape)
            rgb_img = (data["jpg"][i].numpy() + 1) / 2
            print(np.max(rgb_img), np.min(rgb_img))
            plt.subplot(1,2,1), plt.imshow(rgb_img)
            condition = data["hint"][i].numpy()
            if condition_type == "segment_human_and_clothes":
                condition = condition.argmax(-1)
            print("condition max min vals: ", np.max(condition), np.min(condition))
            plt.subplot(1,2,2), plt.imshow(condition)
            plt.title(data["txt"][i])
            # plt.savefig(f"out_{j}_{i}.png")
            plt.show()

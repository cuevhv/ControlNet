import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append(".")

from dataloaders.dataset_bedlam import BedlamSimpleDataset

if __name__ == "__main__":
    plot_imgs = False


    condition_type = ["segment_human_and_clothes", "body_correspondence", "ldmks2d"]
    dataset = BedlamSimpleDataset("dataset/20230804_1_3000_hdri/prompt_train.json", condition_type)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=16, shuffle=True)
    print("dataset images: ", len(dataset))
    print("dataloader length: ", len(dataloader))
    print("batch size: ", dataloader.batch_size)
    print("dataloader length*batch_size: ", dataloader.batch_size*len(dataloader))
    for j, data in tqdm(enumerate(dataloader)):
        for i in range(len(data["txt"])):
            print(data.keys(), data["jpg"].shape, data["hint"].shape)
            rgb_img = (data["jpg"][i].numpy() + 1) / 2
            print(np.max(rgb_img), np.min(rgb_img))
            condition = data["hint"][i].numpy()

            condition_names = data["condition_names"][i].split(',')
            condition_idx = data

            for cond_idx, condition_name in enumerate(condition_names):
                condition_size = data["condition_sizes"][i][cond_idx]
                print("condition name: ", condition_name, condition_size)
                if cond_idx == 0:
                    start_idx = 0
                    end_idx = 0
                else:
                    start_idx = end_idx
                end_idx = end_idx + condition_size
                print("start, end idx",start_idx, end_idx)
                individial_condition_img = condition[..., start_idx:end_idx]
                if condition_name in ["ldmks2d", "segment_human_and_clothes"]:
                    individial_condition_img = individial_condition_img.argmax(-1)
                print("condition max min vals: ", np.max(individial_condition_img), np.min(individial_condition_img))
                if plot_imgs:
                    plt.subplot(1,2,1), plt.imshow(rgb_img)
                    plt.subplot(1,2,2), plt.imshow(individial_condition_img)
                    plt.title(data["txt"][i])
                    # plt.savefig(f"out_{j}_{i}.png")
                    plt.show()

import json
import math
import os
import os.path as osp
import time
from datetime import timedelta

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from base_code.dataset import SceneTextDataset
from base_code.east_dataset import EASTDataset
from base_code.model import EAST


def get_config():
    with open("base_config.json", "r") as file:
        jsonDict = json.load(file)

    if jsonDict["changeable"]["crop_size"] % 32 != 0:
        raise ValueError("`crop_size` must be a multiple of 32")
    return jsonDict


def training(cfg):
    dataset = SceneTextDataset(
        cfg["data_dir"],
        split="train",
        image_size=cfg["changeable"]["image_size"],
        crop_size=cfg["changeable"]["crop_size"],
        ignore_tags=cfg["ignore_tags"],
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / cfg["batch_size"])
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["changeable"]["learning_rate"]
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[cfg["changeable"]["epochs"] // 2], gamma=0.1
    )

    # 이미 있다면 실수로 지워지는 것을 방지하기 위해 error가 뜸
    save_path = osp.join(cfg["save_dir"], cfg["changeable"]["exp_name"])
    os.makedirs(save_path)
    with open(save_path + "/config.json", "w") as file:
        json.dump(cfg, file, indent=4)

    # wandb 설정

    wandb.init(
        entity="hype-squad",
        project="Data_Centric",
        name=cfg["changeable"]["exp_name"],
        config=cfg,
        reinit=True,
    )

    model.train()
    for epoch in range(cfg["changeable"]["epochs"]):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description("[Epoch {}]".format(epoch + 1))

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)
                wandb.log(val_dict)

        scheduler.step()
        wandb.log(
            {
                "Mean loss": epoch_loss / num_batches,
                "Learning Rate": scheduler.get_last_lr(),
            }
        )
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        if (epoch + 1) % cfg["save_interval"] == 0:
            ckpt_fpath = osp.join(save_path, "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)

    wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    training(cfg)

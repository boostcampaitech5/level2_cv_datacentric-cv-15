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


def training(cfg, validation=True):
    def validating(cfg):
        model.eval()

        total_loss = 0
        epoch_start = time.time()
        with tqdm(total=val_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pbar.set_description("[Validation Epoch {}]".format(epoch + 1))

                with torch.no_grad():
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )

                total_loss += loss.item()

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)
                wandb.log(val_dict)
        wandb.log(
            {
                "Mean loss": total_loss / val_num_batches,
            }
        )
        print(
            "Validation Mean loss: {:.4f} | Elapsed time: {}".format(
                total_loss / val_num_batches,
                timedelta(seconds=time.time() - epoch_start),
            )
        )

        return (
            total_loss / val_num_batches,
            extra_info["cls_loss"],
            extra_info["angle_loss"],
            extra_info["iou_loss"],
        )

    dataset = SceneTextDataset(
        cfg["data_dir"],
        split="train1",
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
    )

    if validation:
        val_dataset = SceneTextDataset(
            cfg["data_dir"],
            split="val1",
            image_size=cfg["changeable"]["image_size"],
            crop_size=cfg["changeable"]["crop_size"],
            ignore_tags=cfg["ignore_tags"],
        )
        val_dataset = EASTDataset(val_dataset)
        val_num_batches = math.ceil(len(val_dataset) / cfg["valid_batch_size"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["valid_batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["changeable"]["learning_rate"]
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

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
    patience = 0
    savedLoss = 0
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
                "Learning Rate": scheduler.get_last_lr()[0],
            }
        )
        print(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        if validation:
            val_mean_loss, val_cls_loss, val_angle_loss, val_iou_loss = validating(cfg)

        # early stop, model 저장
        if epoch == 0:
            savedLoss = epoch_loss / num_batches
        else:
            if savedLoss < epoch_loss / num_batches:
                patience += 1
            else:
                patience = 0
                ckpt_fpath = osp.join(save_path, "latest.pth")
                torch.save(model.state_dict(), ckpt_fpath)
                print("new model saved!!!")
        if patience >= 10:
            break
        savedLoss = epoch_loss / num_batches

    wandb.finish()


if __name__ == "__main__":
    cfg = get_config()
    training(cfg)

import numpy as np
import random
import wandb
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LunaSegDataset, LunaSegDatasetTrain
from model import UNetWrapper
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *
from config import Cfg


def train_one_epoch(data, label, model):
    output_prob = model(data)
    d_loss = dice_loss(output_prob, label)
    f_loss = dice_loss(output_prob * label, label)
    loss = d_loss.mean() + f_loss.mean() * 8
    loss.backward()
    return output_prob, loss


@torch.no_grad()
def val_one_epoch(data, label, model):
    output_prob = model(data)
    d_loss = dice_loss(output_prob, label)
    f_loss = dice_loss(output_prob * label, label)
    loss = d_loss.mean() + f_loss.mean() * 8
    return output_prob, loss


def main(cfg):

    # seed point setting
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # dataset
    train_dataloader = DataLoader(
        LunaSegDatasetTrain(data_root=cfg.data_root, val_stride=10),
        batch_size=cfg.bacth_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        LunaSegDataset(data_root=cfg.data_root, val_stride=10, split='val', context_slices_count=3),
        batch_size=cfg.bacth_size,
        num_workers=8,
        shuffle=False,
        pin_memory=True
    )

    # gpu setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = UNetWrapper(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True, up_mode='upconv')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.device_id)

    # optimizer
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)

    # training begins
    min_val_loss = 1000
    for epoch in range(cfg.max_epochs):

        # train one epoch
        train_loss_list = []
        train_accuracy_list, train_precision_list, train_recall_list, train_f1_list = [], [], [], []
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            data, label, _, _ = batch
            data = data.to(device)
            label = label.to(device)
            opt.zero_grad()
            output_prob, train_loss = train_one_epoch(data, label, model)
            opt.step()
            scheduler.step()
            train_loss_list.append(train_loss)

            # compute metrics
            logist = output_prob.argmax(1)
            train_accuracy_list.append(get_accuracy(logist, label))
            train_precision_list.append(get_precision(logist, label))
            train_recall_list.append(get_recall(logist, label))
            train_f1_list.append(get_f1(logist, label))
        wandb.log(
            {'train loss of one epoch': np.mean(train_loss_list)},
            {'training accuracy of one epoch': np.mean(train_accuracy_list)},
            {'training precision of one epoch': np.mean(train_precision_list)},
            {'training recall of one epoch': np.mean(train_recall_list)},
            {'training f1 of one epoch': np.mean(train_f1_list)},
            {'epoch': epoch}
        )

        # validate one epoch
        val_loss_list = []
        val_accuracy_list, val_precision_list, val_recall_list, val_f1_list = [], [], [], []
        model.eval()
        for batch in val_dataloader:
            data, label, _, _ = batch
            data = data.to(device)
            label = label.to(device)
            output_prob, val_loss = val_one_epoch(data, label, model)
            val_loss_list.append(val_loss)

            # compute metrics
            logist = output_prob.argmax(1)
            val_accuracy_list.append(get_accuracy(logist, label))
            val_precision_list.append(get_precision(logist, label))
            val_recall_list.append(get_recall(logist, label))
            val_f1_list.append(get_f1(logist, label))
        wandb.log(
            {'val loss of one epoch': np.mean(train_loss_list)},
            {'val accuracy of one epoch': np.mean(val_accuracy_list)},
            {'val precision of one epoch': np.mean(val_precision_list)},
            {'val recall of one epoch': np.mean(val_recall_list)},
            {'val f1 of one epoch': np.mean(val_f1_list)},
            {'epoch': epoch}
        )

        #  save the model whose val loss is minimum
        if np.mean(val_loss_list) < min_val_loss:
            min_val_loss = np.mean(val_loss_list)
            os.makedirs(os.path.join(cfg.save_path, cfg.run_name), exist_ok=True)
            if torch.cuda.device_count() > 1:
                torch.save(
                    model.module.state_dict(),
                    os.path.join(cfg.save_path, cfg.run_name, 'model.h5')
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_path, cfg.run_name, 'model.h5')
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/baseline.yaml")
    parser.add_argument("--debug", help='whether debug', action='store_true')
    args = parser.parse_args()

    config = Cfg(cfg_file=args.cfg_file)
    cfg = config.get_cfg()
    if args.debug:
        main(cfg)
    else:
        wandb.init(name=cfg.run_name, project=cfg.project_name)
        wandb.config.update(cfg)
        main(wandb.config)





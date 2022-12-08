import torch.cuda
import numpy as np
import random
import wandb
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LunaSegDataset, LunaSegDatasetTrain
from model import UNetWrapper
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


def train_one_epoch(data, label, model, criterion):
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    return loss.item()


@torch.no_grad()
def val_one_epoch(data, label, model, criterion):
    output = model(data)
    loss = criterion(output, label)
    return loss.item()


def main(cfg):

    # seed point setting
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # dataset
    train_dataloader = DataLoader(
        LunaSegDatasetTrain(
            data_root=cfg.data_root,
            val_stride=10
        ),
        batch_size=cfg.bacth_size,
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        LunaSegDataset(
            data_root=cfg.data_root,
            val_stride=10,
            split='val',
            context_slices_count=3,
        ),
        batch_size=cfg.bacth_size,
        shuffle=False,
        pin_memory=True
    )

    # gpu setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = UNetWrapper().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=cfg.device_id)

    # optimizer
    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    weight = torch.tensor([cfg.CEWeight, 1 - cfg.CEWeight], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weight).to(device)

    # train begins
    min_val_loss = 1000
    for epoch in range(cfg.max_epochs):

        # train one epoch
        train_loss_list = []
        model.train()
        for batch in train_dataloader:
            data, label, _, _ = batch
            data = data.to(device)
            label = label.to(torch.long)
            opt.zero_grad()
            train_loss = train_one_epoch(data, label, model, criterion)
            opt.step()
            scheduler.step()
            train_loss_list.append(train_loss)
        wandb.log(
            {'train loss of one epoch': np.mean(train_loss_list)},
            {'epoch': epoch}
        )

        # validate one epoch
        val_loss_list = []
        model.eval()
        for batch in val_dataloader:
            data, label, _, _ = batch
            data = data.to(device)
            label = label.to(torch.long)
            val_loss = train_one_epoch(data, label, model, criterion)
            val_loss_list.append(val_loss)
        wandb.log(
            {'val loss of one epoch': np.mean(train_loss_list)},
            {'epoch': epoch}
        )

        #  save model whose val loss is minimum
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            os.makedirs(os.path.join(cfg.save_path, cfg.run_name), exist_ok=True)
            if torch.cuda.device_count() > 1:
                torch.save(
                    model.module.state_dict(),
                    model.state_dict(),
                    os.path.join(cfg.save_path, cfg.run_name, 'model.h5')
                )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.save_path, cfg.run_name, 'model.h5')
                )






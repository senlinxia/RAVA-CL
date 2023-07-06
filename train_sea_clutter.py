from config import cfg
import os
import torch
import torch.nn as nn
from sea_clutter_dataloader import SeaClutterDataset
import random
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
from networks.ResNetRAVA_CL import ResNetRAVA_CL
import torch.nn.functional as F


def info_nce_loss( features):
    labels = torch.cat(
        [torch.arange(features.shape[0]//2) for i in range(2)],
        dim=0)

    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.cuda()

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)  # 512x128 128x512


    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = similarity_matrix[~mask].view(
        similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(
        similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    logits = logits / cfg.SEA_CLUTTER.TEMPERATURE

    return logits, labels

def train(distribution=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE

    dataset = SeaClutterDataset(whole=True, distribution=distribution)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  sampler=sampler.SubsetRandomSampler(
                                      indices[
                                      :int(1 * len(dataset))]))

    model=ResNetRAVA_CL(cfg.SEA_CLUTTER.FEATURE_DIM)
    model=model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), cfg.TRAIN.LR,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=0,last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000,
                                                gamma=0.8)
    criterion=nn.CrossEntropyLoss()

    step=0
    train_step=500

    for epoch in range(cfg.TRAIN.EPOCH):


        for i, (datapack) in enumerate(train_dataloader):
            model.train()
            echo=datapack['echos']
            echo=torch.cat(echo, dim=0)
            echo=echo.cuda().float()
            features = model(echo)

            logits, labels = info_nce_loss(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            print(loss.item())

            if (step) % train_step == 0:
                model.eval()

                if distribution:
                    pt_dir = os.path.join(cfg.TRAIN.TRAINED_PARAMETER_PATH,
                                      '%d_%d_%d_sea_clutter_%s' % (
                                          cfg.SEA_CLUTTER.SNR_MIN,
                                          cfg.SEA_CLUTTER.SNR_MAX,
                                          cfg.SEA_CLUTTER.SMALL_LEN,distribution))
                else:
                    pt_dir = os.path.join(cfg.TRAIN.TRAINED_PARAMETER_PATH,
                                          '%d_%d_%d_sea_clutter' % (
                                              cfg.SEA_CLUTTER.SNR_MIN,
                                              cfg.SEA_CLUTTER.SNR_MAX,
                                              cfg.SEA_CLUTTER.SMALL_LEN))

                if not os.path.isdir(pt_dir):
                    os.makedirs(pt_dir)

                torch.save(model.state_dict(),
                           os.path.join(
                               pt_dir,
                               '%d_%d_%d_sea_clutter_ravacl_%d.pt' % (
                               cfg.SEA_CLUTTER.SNR_MIN,
                               cfg.SEA_CLUTTER.SNR_MAX,
                               cfg.SEA_CLUTTER.SMALL_LEN, epoch)))

if __name__ == '__main__':
    train()




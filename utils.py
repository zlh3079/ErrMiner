import random

import os
import numpy as np
import torch

from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader

from tqdm import tqdm
import torch.optim as optim
from dataloader import dialogDataset


def seed_everything(seed=2112):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_dialog_loaders(train_address, test_address, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):
    train_set = dialogDataset(train_address, pretrained_model)

    test_set = dialogDataset(test_address, pretrained_model)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_model(model, loss_func, dataloader, optimizer, epoch, cuda, save_path, seed=2112):
    losses, preds, labels, ids = [], [], [], []

    model.train()

    seed_everything(seed + epoch)
    for data in tqdm(dataloader):
        # clear the grad
        optimizer.zero_grad()

        input_ids, token_type_ids, attention_mask_ids, graph_label = \
            [d.cuda() for d in data[:-4]] if cuda else data[:-4]
        dialog_id, role, graph_edge, seq_len = data[-4:]

        log_prob = model(input_ids, token_type_ids,
                         attention_mask_ids, role, seq_len, graph_edge)
        loss = loss_func(log_prob, graph_label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(graph_label.cpu().numpy())
        losses.append(loss.item())
        ids += dialog_id

        # accumulate the grad
        loss.backward()
        # optimizer the parameters
        optimizer.step()

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    torch.save(model.state_dict(), save_path)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


def meta_train(model, loss_func, dataloader, tar_loader, optimizer, epoch, cuda, save_path, seed=2112):
    losses, preds, labels, ids = [], [], [], []
    model_copy = model
    model_copy.train()
    optimizer_copy = optim.Adam(filter(lambda p: p.requires_grad, model_copy.parameters()))
    seed_everything(seed + epoch)
    for data in tqdm(dataloader):
        # clear the grad
        optimizer_copy.zero_grad()

        input_ids, token_type_ids, attention_mask_ids, graph_label = \
            [d.cuda() for d in data[:-4]] if cuda else data[:-4]
        dialog_id, role, graph_edge, seq_len = data[-4:]

        log_prob = model_copy(input_ids, token_type_ids,
                              attention_mask_ids, role, seq_len, graph_edge)
        loss = loss_func(log_prob, graph_label)

        # accumulate the grad
        loss.backward()
        # optimizer the parameters
        optimizer_copy.step()

    for data in tqdm(tar_loader):
        # clear the grad
        optimizer.zero_grad()

        input_ids, token_type_ids, attention_mask_ids, graph_label = \
            [d.cuda() for d in data[:-4]] if cuda else data[:-4]
        dialog_id, role, graph_edge, seq_len = data[-4:]

        log_prob = model_copy(input_ids, token_type_ids,
                              attention_mask_ids, role, seq_len, graph_edge)
        loss = loss_func(log_prob, graph_label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(graph_label.cpu().numpy())
        losses.append(loss.item())
        ids += dialog_id

        # accumulate the grad
        loss.backward()
        # optimizer the parameters
        optimizer.step()

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    torch.save(model.state_dict(), save_path)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


def evaluate_model(model, loss_func, dataloader, cuda):
    losses, preds, labels, ids = [], [], [], []

    model.eval()

    seed_everything()
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask_ids, graph_label = \
                [d.cuda() for d in data[:-4]] if cuda else data[:-4]
            dialog_id, role, graph_edge, seq_len = data[-4:]

            log_prob = model(input_ids, token_type_ids,
                             attention_mask_ids, role, seq_len, graph_edge)
            loss = loss_func(log_prob, graph_label)

            preds.append(torch.argmax(log_prob, 1).cpu().numpy())
            labels.append(graph_label.cpu().numpy())
            losses.append(loss.item())
            ids += dialog_id

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


class Config(object):
    def __init__(self):
        self.cuda = True
        # self.train_address = f"./data/augmented/{project}"  # angular, appium, docker, dl4j, gitter, typescript
        # self.test_address = f"./data/processed/{project}"
        self.pretrained_model = './pretrained_model/bert_base_uncased'
        self.D_bert = 768
        self.filter_sizes = [2, 3, 4, 5]
        self.filter_num = 50
        self.D_cnn = 100
        self.D_graph = 64
        self.lr = 1e-4
        self.l2 = 1e-5
        self.batch_size = 64
        self.graph_class_num = 2
        self.dropout = 0.5
        self.epochs = 50

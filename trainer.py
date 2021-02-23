import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

from tqdm import tqdm

def train(args, trn_cfg):
    
    train_loader = trn_cfg['train_loader']
    valid_loader = trn_cfg['valid_loader']
    model = trn_cfg['model']
    criterion = trn_cfg['criterion']
    optimizer = trn_cfg['optimizer']
    scheduler = trn_cfg['scheduler']
    device = trn_cfg['device']
    fold_num = trn_cfg['fold_num']

    best_epoch = 0
    best_val_score = 0.0

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        
        start_time = time.time()
    
        trn_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_score = validation(args, trn_cfg, model, criterion, valid_loader, device)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - trn_loss: {:.4f}  val_loss: {:.4f}  val_acc: {:.4f}, val_score_f1: {:.4f} lr: {:.5f}  time: {:.0f}s\n".format(
                epoch+1, trn_loss, val_loss, val_acc, val_score, lr[0], elapsed))

        
        # save model weight
        if val_score > best_val_score:
            best_val_score = val_score            
            file_save_name = 'best_score' + '_fold' + str(fold_num)

        if args.scheduler == 'Plateau':
            scheduler.step(val_score)
        else:
            scheduler.step()
    

def train_one_epoch(args, model, criterion, train_loader, optimizer, scheduler, device):

    model.train()
    trn_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(train_loader):
        loss=0.0
        if device:
            images = images.to(device)
            labels = labels.long().to(device)
        
        # Forward pass
        outputs = model(images)
        outputs = torch.split(outputs, [2 for _ in range(26)], dim=1)
        
        for i in range(26):
            loss+=criterion(outputs[i], labels[:, i])
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()

    epoch_train_loss = trn_loss / len(train_loader)

    return epoch_train_loss


def validation(args, trn_cfg, model, criterion, valid_loader, device):
    
    model.eval()
    val_loss = 0.0
    total_labels = []
    total_outputs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_loader):
            loss=0.0
            total_labels.append(labels)

            if device:
                images = images.to(device)
                labels = labels.long().to(device)

            outputs = torch.sigmoid(model(images))
            # outputs = model(images)
            outputs = torch.split(outputs, [2 for _ in range(26)], dim=1)
            for i in range(26):
                loss+=criterion(outputs[i], labels[:, i])
            # loss = criterion(outputs, labels)

            val_loss += loss.item()
            total_outputs.append([o.cpu().detach().numpy() for o in outputs])

    epoch_val_loss = val_loss / len(valid_loader)

    total_labels = np.concatenate(np.concatenate(total_labels)).tolist()
    total_outputs = np.concatenate(np.concatenate(total_outputs)).tolist()
    total_outputs = np.argmax(total_outputs, 1)

    acc = accuracy_score(total_labels, total_outputs)
    metrics = f1_score(total_labels, total_outputs)
    acc, metrics = np.round(acc, 4), np.round(metrics, 4)

    # val_score = np.round(np.mean(list(metrics.values())), 4)
    val_acc = acc
    val_score = metrics
    
    return epoch_val_loss, val_acc, val_score


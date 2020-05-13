import gc
import os
import time
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conf import config
from model.net import Net
from utils.data_utils import MyDataset
from utils.data_utils import train_transform, val_transform, test_transform
from utils.model_utils import accuracy, FocalLoss
from utils.utils import AverageMeter, ProgressMeter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, val_loader, criterion):
    # switch to evaluate mode
    model.eval()
    data_len = 0
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_len = len(batch_y)
            # batch_len = len(batch_y.size(0))
            data_len += batch_len
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x)
            loss = criterion(probs, batch_y)
            total_loss += loss.item()
            _acc = accuracy(probs, batch_y)
            total_acc += _acc[0].item() * batch_len

    return total_loss / data_len, total_acc / data_len


def train(train_data, val_data, fold_idx=None):
    train_data = MyDataset(train_data, train_transform)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = MyDataset(val_data, val_transform)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    model = Net(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # config.model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))

    best_val_acc = 0
    last_improved_epoch = 0
    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch: ', cur_epoch + 1)
        cur_step = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            probs = model(batch_x)

            train_loss = criterion(probs, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.train_print_step == 0:
                train_acc = accuracy(probs, batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                print(msg.format(cur_step, len(train_loader), train_loss.item(), train_acc[0].item()))
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        # msg = 'the current epoch: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%},  ' \
        #       'val loss: {4:>5.2}, val acc: {5:>6.2%}, {6}'
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_acc,
                         end_time - start_time, improved_str))
        scheduler.step()
        if cur_epoch - last_improved_epoch > config.patience_epoch:
            print("No optimization for a long time, auto-stopping...")
            break
    del model
    gc.collect()


def predict():
    model = Net(model_name).to(device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))

#    data_len = len(os.listdir(config.image_test_path))
#    test_path_list = ['{}/{}.jpg'.format(config.image_test_path, x) for x in range(0, data_len)]
#    test_data = np.array(test_path_list)
    test_df = pd.read_csv(config.test_path)
    test_df['FileID'] = test_df['FileID'].apply(lambda x: '{}/{}.jpg'.format(config.image_test_path, x))
    print('test:{}'.format(test_df.shape[0]))
    test_dataset = MyDataset(test_df, test_transform, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            # compute output
            probs = model(batch_x)
            preds = torch.argmax(probs, dim=1)
            pred_list += [p.item() for p in preds]

    submission = pd.DataFrame({"FileID": range(len(pred_list)), "SpeciesID": pred_list})
    submission.to_csv('submission.csv', index=False, header=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        print(train_df['SpeciesID'].value_counts())
        train_df['FileID'] = train_df['FileID'].apply(lambda x: '{}/{}.jpg'.format(config.image_train_path, x))
        if mode == 1:
            train_data, val_data = train_test_split(train_df, shuffle=True, test_size=0.1)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
        else:
            n_splits = 5
            x = train_df['FileID'].values
            y = train_df['SpeciesID'].values
            skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=16, type=int, help="train epochs")
    parser.add_argument("-m", "--model_name", default='resnet', type=str, help="model select")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()
    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model_name
    mode = args.mode

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args.operation)

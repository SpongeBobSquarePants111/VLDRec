import logging
import argparse
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import random

from sklearn.preprocessing import LabelEncoder

import torch
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from utils import *
from models import *
from datasets import *


def train_epoch(train_loader, model, optimizer, gpu_id, alpha):
    model.train()
    loss_sum = []
    loss1_sum = []
    loss2_sum = []

    if alpha != 0:
        for input in tqdm(train_loader):
            recursive_to(input, 'cuda:%d' % gpu_id)
            optimizer.zero_grad()

            output1_task1, output2_task1, output1_task2, output2_task2 = model(input, 1)
            loss1 = torch.mean(torch.log(1 + torch.exp(output2_task1 - output1_task1 - 1e-8)))
            loss2 = torch.mean(torch.log(1 + torch.exp(output2_task2 - output1_task2 - 1e-8)))
            loss = (1-alpha) * loss1 + alpha * loss2

            if torch.isnan(loss):
                print('Loss was NaN')
                logging.info('Loss was NaN')
                break
            loss.backward()
            loss_sum.append(float(loss.cpu()))
            loss1_sum.append(float(loss1.cpu()))
            loss2_sum.append(float(loss2.cpu()))

            optimizer.step()
        return sum(loss1_sum) / len(train_loader), sum(loss2_sum) / len(train_loader), \
               sum(loss_sum) / len(train_loader)
    else:
        for input in tqdm(train_loader):
            recursive_to(input, 'cuda:%d' % gpu_id)
            optimizer.zero_grad()
            output1_task1, output2_task1 = model(input, 2)
            loss = torch.mean(torch.log(1 + torch.exp(output2_task1 - output1_task1 - 1e-8)))

            if torch.isnan(loss):
                print('Loss was NaN')
                logging.info('Loss was NaN')
                pdb.set_trace()
            loss.backward()
            loss_sum.append(float(loss.cpu()))
            optimizer.step()
        return sum(loss_sum) / len(train_loader), 0, sum(loss_sum) / len(train_loader)


def get_predict(data_loader, model, gpu_id):
    model.eval()
    with torch.autograd.no_grad():
        output_all = []
        for input in data_loader:
            recursive_to(input, 'cuda:%d' % gpu_id)
            output = model(input, 0)
            output = torch.reshape(output, [-1, 1])
            output_all += [np.array(output.cpu())]
        output_all = np.reshape(np.concatenate(output_all, axis=0), [-1, 1])

        return output_all


def train_model(data_loader, duration_time_test, duration_time_neg_data, user_list_neg,
                rank_list, model, log_dir_exp, args):

    train_loader, val_loader, test_loader, neg_data_valid_loader, neg_data_test_loader = data_loader
    best_metric = -1
    best_epoch_metric = [-1, -1, -1]
    best_test_predict = [-1, -1]
    best_rec_res = None
    early_stop_count = 0
    top_time = [120, 240]
    model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss1, train_loss2, train_loss = train_epoch(train_loader, model, optimizer, args.gpu_id, args.alpha)

        neg_score_valid = np.reshape(get_predict(neg_data_valid_loader, model, args.gpu_id), [-1, 100])
        neg_score_test = np.reshape(get_predict(neg_data_test_loader, model, args.gpu_id), [-1, 100])
        val_predict = np.reshape(get_predict(val_loader, model, args.gpu_id), [-1])
        test_predict = np.reshape(get_predict(test_loader, model, args.gpu_id), [-1])

        val_view_time_t, val_gauc, val_auc, _ = \
            get_performance(val_predict, neg_score_valid, duration_time_test, duration_time_neg_data,
                            user_list_neg[0], rank_list[0], top_time)

        test_view_time_t, test_gauc, test_auc, test_rec_res = \
            get_performance(test_predict, neg_score_test, duration_time_test, duration_time_neg_data,
                            user_list_neg[1], rank_list[1], top_time)

        log_text = '[epoch %d]:  train_loss1 = %.4f, train_loss2 = %.4f, train_loss =  %.4f, \n' % \
                   (epoch, train_loss1, train_loss2, train_loss)
        get_log_text(log_text, top_time, [val_view_time_t, val_gauc, val_auc], 'val')

        log_text = ''
        get_log_text(log_text, top_time, [test_view_time_t, test_gauc, test_auc], 'test')

        val_metric = val_auc[0]
        if val_metric > best_metric:
            torch.save(model, os.path.join(log_dir_exp, 'model.ckpt'))
            best_metric = val_metric
            best_rec_res = test_rec_res
            best_epoch_metric = [test_view_time_t, test_gauc, test_auc]
            best_test_predict = [test_predict, neg_score_test]
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= args.early_stop and epoch >= args.min_epoch:
                print('early stop!')
                break

    print('-----Result-----')
    logging.info('-----Result-----')

    log_text = 'best_epoch, '
    get_log_text(log_text, top_time, best_epoch_metric, 'test')

    pickle.dump(best_epoch_metric, open(os.path.join(log_dir_exp, 'performance_metric.pkl'), 'wb'))
    pickle.dump(best_rec_res, open(os.path.join(log_dir_exp, 'rec_res.pkl'), 'wb'))
    np.savetxt(os.path.join(log_dir_exp, 'test_result.csv'), best_test_predict[0])
    np.savetxt(os.path.join(log_dir_exp, 'test_neg_result.csv'), best_test_predict[1])


def load_data(args):
    data_train = pickle.load(open(os.path.join(args.data_dir, 'train.pkl'), 'rb'))
    data_valid = pickle.load(open(os.path.join(args.data_dir, 'valid.pkl'), 'rb'))
    data_test = pickle.load(open(os.path.join(args.data_dir, 'test.pkl'), 'rb'))
    neg_data_valid = pickle.load(open(os.path.join(args.data_dir, 'neg_data_valid.pkl'), 'rb'))
    neg_data_test = pickle.load(open(os.path.join(args.data_dir, 'neg_data_test.pkl'), 'rb'))

    user_video_list1 = pickle.load(open(os.path.join(args.data_dir, 'user_video_list1.pkl'), 'rb'))
    user_video_list2 = pickle.load(open(os.path.join(args.data_dir, 'user_video_list2.pkl'), 'rb'))
    user_video_list = [user_video_list1, user_video_list2]

    user_sample_dict = pickle.load(open(os.path.join(args.data_dir ,'user_sample_dict.pkl'), 'rb'))

    duration_time_test = np.array(data_test['duration_time'])
    duration_time_neg_data = np.reshape(np.array(neg_data_test['duration_time']), [-1, 100])

    valid_user_list_neg = list(neg_data_valid['user_id'])
    valid_user_list_neg = [valid_user_list_neg[i*100] for i in range(int(len(valid_user_list_neg) / 100))]
    test_user_list_neg = list(neg_data_test['user_id'])
    test_user_list_neg = [test_user_list_neg[i * 100] for i in range(int(len(test_user_list_neg) / 100))]
    user_list_neg = [valid_user_list_neg, test_user_list_neg]

    cat_feat_name = ['user_id', 'video_id', 'duration_time_discrete']

    data_all = pd.concat([data_train, data_valid, data_test, neg_data_valid, neg_data_test], axis=0)

    print('Feature transform')
    lbe = LabelEncoder()
    data_all['video_id'] = lbe.fit_transform(data_all['video_id'].astype(str))

    num_embeddings = get_cat_num_embeddings(data_all, cat_feat_name)

    data_train = data_all.iloc[0:data_train.shape[0], :]
    data_valid = data_all.iloc[data_train.shape[0]: data_train.shape[0] + data_valid.shape[0], :]
    data_test = data_all.iloc[data_train.shape[0] + data_valid.shape[0]:
                              data_train.shape[0] + data_valid.shape[0] + data_test.shape[0], :]
    neg_data_valid = data_all.iloc[data_train.shape[0] + data_valid.shape[0] + data_test.shape[0]:
                                   data_train.shape[0] + data_valid.shape[0] + data_test.shape[0] +
                                   neg_data_valid.shape[0], :]
    neg_data_test = data_all.iloc[data_train.shape[0] + data_valid.shape[0] + data_test.shape[0]
                                  + neg_data_valid.shape[0]:, :]

    train_dataset = TrainDataset(data_train, cat_feat_name, user_video_list, user_sample_dict, args)
    val_dataset = TestDataset(data_valid, cat_feat_name)
    test_dataset = TestDataset(data_test, cat_feat_name)
    valid_neg_dataset = TestDataset(neg_data_valid, cat_feat_name)
    test_neg_dataset = TestDataset(neg_data_test, cat_feat_name)

    collate_fn_train1 = get_collator_train1()
    collate_fn_train2 = get_collator_train2()
    collate_fn_val_test = get_collator_val_test()

    if args.alpha != 0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                                  collate_fn=collate_fn_train1, shuffle=True, pin_memory=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                                  collate_fn=collate_fn_train2, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                            collate_fn=collate_fn_val_test, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.data_loader_workers,
                             collate_fn=collate_fn_val_test, pin_memory=False)
    neg_data_valid_loader = DataLoader(valid_neg_dataset, batch_size=args.batch_size,
                                       num_workers=args.data_loader_workers,
                                       collate_fn=collate_fn_val_test, pin_memory=False)
    neg_data_test_loader = DataLoader(test_neg_dataset, batch_size=args.batch_size,
                                      num_workers=args.data_loader_workers,
                                      collate_fn=collate_fn_val_test, pin_memory=False)

    rank_list = pickle.load(open(os.path.join(args.data_dir, 'rank_list.pkl'), 'rb'))
    data_loader = (train_loader, val_loader, test_loader, neg_data_valid_loader, neg_data_test_loader)

    return data_loader, duration_time_test, duration_time_neg_data, num_embeddings, user_list_neg, rank_list


def parse_args():
    parser = argparse.ArgumentParser(description="Run DNN")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--embedding_size', type=int, default=8)
    parser.add_argument('--early_stop',type=int, default=10, help='early stop patience')
    parser.add_argument('--min_epoch', type=int, default=10)
    parser.add_argument('--data_loader_workers', type=int, default=15)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--log_dir', type=str, default='./log')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dnn_hidden_units', type=str, default='[32, 16]')
    parser.add_argument('--trial', type=str, default='0',help='Indicate trail id with same condition')
    parser.add_argument('--fix_seed', action='store_true', default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.fix_seed:
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        print("Fix Random Seed!!!")
    torch.cuda.set_device(args.gpu_id)
    args.dnn_hidden_units = eval(args.dnn_hidden_units)

    log_dir_exp = init_log(args)
    data_loader, duration_time_test, duration_time_neg_data, num_embeddings, \
    user_list_neg, rank_list = load_data(args)

    model = BaselineModel(num_embeddings, args)
    train_model(data_loader, duration_time_test, duration_time_neg_data, user_list_neg,
                rank_list, model, log_dir_exp, args)



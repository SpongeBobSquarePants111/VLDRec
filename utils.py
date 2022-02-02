import logging
import math
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pdb
import pickle
import torch
import torch.multiprocessing
import copy


def init_log(args):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir_exp = os.path.join(args.log_dir, 'log-%s-lr%.4f-do%.2f-emb%d-beta%.2f-%s-alpha%.2f-earlystop%d-%s' %
                               (log_time, args.lr, args.dropout, args.embedding_size, args.beta,
                                '['+','.join([str(s) for s in args.dnn_hidden_units])+']',
                                args.alpha, args.early_stop, args.trial))

    if args.fix_seed:
        log_dir_exp += '-fix_seed'

    print(log_dir_exp)
    if not os.path.exists(log_dir_exp):
        os.mkdir(log_dir_exp)
    logging.basicConfig(filename=os.path.join(log_dir_exp, 'log'), level=logging.INFO, filemode="w")
    return log_dir_exp


def recursive_to(iterable, device):
    if isinstance(iterable, torch.Tensor):
        iterable.data = iterable.data.to(device)
    elif isinstance(iterable, (list, tuple)):
        for v in iterable:
            recursive_to(v, device)


def get_cat_num_embeddings(data, cat_feat_name):
    cat_num_embeddings = []
    for each in cat_feat_name:
        cat_num_embeddings.append(np.max(np.array((data[each])))+1)
    return cat_num_embeddings


def cal_view_time_with_neg(predict_prob, neg_score_data, duration_time_test, duration_time_neg_data,
                           rank_list, top_time, user_id_to_idx):
    view_time_all_user_t = []

    rec_res_t = [[], []]

    for uid in rank_list.keys(): 
        if len(rank_list[uid]) <= 3:
            continue
        score_all = []
        for each in rank_list[uid]:  
            score_all.append([predict_prob[each[0]], duration_time_test[each[0]], each[1], 1])
        user_idx = user_id_to_idx[uid]
        for j in range(neg_score_data.shape[1]):
            score_all.append([neg_score_data[user_idx, j], duration_time_neg_data[user_idx, j], 0, 0])
        score_all = sorted(score_all, key=lambda x: x[0], reverse=True)

        sum_now = 0
        for j in range(len(top_time)):
            rec_res_t[j].append([])
        view_time_one_user_t = [0] * len(top_time)
        for i in range(len(score_all)):
            for j in range(len(top_time)):
                if sum_now > top_time[j]:
                    continue
                elif sum_now + score_all[i][1] > top_time[j]:
                    if score_all[i][3] == 1:
                        view_time_one_user_t[j] += score_all[i][2]*((top_time[j] - sum_now)/score_all[i][1])
                    rec_res_t[j][-1].append(score_all[i][1])
                else:
                    if score_all[i][3] == 1:
                        view_time_one_user_t[j] += score_all[i][2]
                    rec_res_t[j][-1].append(score_all[i][1])
            sum_now += score_all[i][1]
        view_time_all_user_t.append(view_time_one_user_t)

    view_time_all_user_t = list(np.mean(np.array(view_time_all_user_t), axis=0))

    return view_time_all_user_t, rec_res_t


def cal_auc_without_neg(predict_prob, rank_list):
    auc = []
    pos_sample_label_all = []
    pos_score_all = []
    for uid in rank_list.keys(): 
        rank_list_one_user = rank_list[uid]

        if len(rank_list_one_user) <= 3:
            continue

        for i in range(len(rank_list_one_user)):
            rank_list_one_user[i].append(predict_prob[rank_list_one_user[i][0]])
        pos_score_one_user = [each[3] for each in rank_list_one_user]
        pos_sample_label = [each[2] for each in rank_list_one_user]

        pos_score_all += pos_score_one_user
        pos_sample_label_all += pos_sample_label

        if sum(pos_sample_label) == 0 or sum(pos_sample_label)==len(pos_sample_label):
            continue
        else:
            scaler = MinMaxScaler()
            pos_score_one_user = scaler.fit_transform(np.reshape(np.array(pos_score_one_user), [-1,1])).tolist()
            try:
                auc.append([roc_auc_score(pos_sample_label, pos_score_one_user), len(pos_sample_label)])
            except:
                pdb.set_trace()

    scaler = MinMaxScaler()
    pos_score_all = scaler.fit_transform(np.reshape(np.array(pos_score_all), [-1, 1])).tolist()
    auc_all = roc_auc_score(pos_sample_label_all, pos_score_all)

    auc_sum = 0
    valid_sample = 0
    for each in auc:
        auc_sum += each[0]*each[1]
        valid_sample += each[1]
    gauc = auc_sum/valid_sample

    return gauc, auc_all


def get_performance(predict_prob, neg_score, duration_time_test, duration_time_neg_data,
                    user_list_neg, rank_list, top_time):
    neg_score = np.reshape(neg_score, [-1, 100])
    assert neg_score.shape[0] == len(user_list_neg)
    user_id_to_idx = {}

    for i in range(len(user_list_neg)):
        user_id_to_idx[user_list_neg[i]] = i

    rank_list_tmp = copy.deepcopy(rank_list[0])
    view_time_t, rec_res_k = cal_view_time_with_neg(predict_prob, neg_score, duration_time_test, duration_time_neg_data,
                               rank_list_tmp, top_time, user_id_to_idx)

    gauc = []
    auc = []
    for i in range(len(rank_list)):
        rank_list_tmp = copy.deepcopy(rank_list[i])
        res = cal_auc_without_neg(predict_prob, rank_list_tmp)
        gauc.append(res[0])
        auc.append(res[1])
    return view_time_t, gauc, auc, rec_res_k


def get_log_text(log_text, top_time, metric, mode):
    view_time_t, gauc, auc = metric

    for i in range(len(top_time)):
        log_text += '%s_view_time_t@%d = %.4f, ' % (mode, top_time[i], view_time_t[i])
    log_text += '\n'

    log_text += '%s_GAUC = %.4f, ' % (mode, gauc[0])
    log_text += '%s_AUC = %.4f, ' % (mode, auc[0])

    log_text += '%s_GAUC_group = [%.4f, %.4f, %.4f, %.4f, %.4f], ' % (mode, gauc[1], gauc[2], gauc[3], gauc[4], gauc[5])
    log_text += '%s_AUC_group = [%.4f, %.4f, %.4f, %.4f, %.4f], ' % (mode, auc[1], auc[2], auc[3], auc[4], auc[5])

    log_text = log_text.strip(', ')
    print(log_text)
    logging.info(log_text)


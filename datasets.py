import numpy as np
import random

import pdb
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, dataset, cat_feat_name, user_video_list, user_sample_dict, args):
        self.data = dataset
        self.data_cat = self.data[cat_feat_name]
        self.alpha = args.alpha
        self.beta = args.beta
        self.user_video_list1 = user_video_list[0]
        self.user_video_list2 = user_video_list[1]
        self.user_sample_dict_pos = user_sample_dict[0]
        self.user_sample_dict_neg = user_sample_dict[1]

        self.y_true = [float(r) for r in list(self.data['progress_rate'])]

        self.user_list = list(self.data['user_id'])
        self.label_list = list(self.data['label'])
        self.duration_time_discrete_list = list(self.data['duration_time_discrete'])


    def sample_based_on_progress_rate(self, item, video_list):
        rand_num = torch.randint(0, len(video_list), (1,)).item()
        pair_id = video_list[rand_num]
        while pair_id == item or abs(self.y_true[item] - self.y_true[pair_id]) < 0.05:
            rand_num = torch.randint(0, len(video_list), (1,)).item()
            pair_id = video_list[rand_num]
        return pair_id

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        cat_feat1 = self.data_cat.iloc[item, :]
        uid = self.user_list[item]
        video_list1 = self.user_video_list1[uid]

        if self.label_list[item] == 1:
            user_sample_dict = self.user_sample_dict_neg
        else:
            user_sample_dict = self.user_sample_dict_pos

        p = torch.rand(1, ).item()
        if p < self.beta or uid not in user_sample_dict[0]:  
            pair_id1 = self.sample_based_on_progress_rate(item, video_list1)
            cat_feat2 = self.data_cat.iloc[pair_id1, :]
            if self.y_true[item] < self.y_true[pair_id1]:
                cat_feat1, cat_feat2 = cat_feat2, cat_feat1
        else:  
            rand_num = torch.randint(0, len(user_sample_dict[0][uid]), (1,)).item()
            pair_id1 = user_sample_dict[0][uid][rand_num]
            cat_feat2 = self.data_cat.iloc[pair_id1, :]
            if self.label_list[item] == 0:
                cat_feat1, cat_feat2 = cat_feat2, cat_feat1

        if self.alpha != 0:
            item2 = item
            cat_feat3 = cat_feat1
            uid2 = self.user_list[item2]

            duration_time_discrete = self.duration_time_discrete_list[item2]
            video_list2 = self.user_video_list2[uid2][int(self.duration_time_discrete_list[item2])]

            if p < self.beta or uid not in user_sample_dict[duration_time_discrete + 1]:
                pair_id2 = self.sample_based_on_progress_rate(item2, video_list2)
                cat_feat4 = self.data_cat.iloc[pair_id2, :]
                if self.y_true[item2] < self.y_true[pair_id2]:
                    cat_feat3, cat_feat4 = cat_feat4, cat_feat3
            else:
                rand_num = torch.randint(0, len(user_sample_dict[duration_time_discrete + 1][uid]), (1,)).item()
                pair_id2 = user_sample_dict[duration_time_discrete + 1][uid][rand_num]
                cat_feat4 = self.data_cat.iloc[pair_id2, :]
                if self.label_list[item] == 0:
                    cat_feat3, cat_feat4 = cat_feat4, cat_feat3

            return (cat_feat1, cat_feat2, cat_feat3, cat_feat4)
        else:
            return (cat_feat1, cat_feat2)


class TestDataset(Dataset):
    def __init__(self, dataset, cat_feat_name):
        self.data = dataset
        self.data_cat = self.data[cat_feat_name]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item: int):
        cat_feat = self.data_cat.iloc[item, :]
        return cat_feat


def get_collator_train1():
    def collator(data_points):
        packed_cat_feat1 = []
        packed_cat_feat2 = []
        packed_cat_feat3 = []
        packed_cat_feat4 = []
        for (cat_feat1, cat_feat2, cat_feat3, cat_feat4) in data_points:
            packed_cat_feat1.append(torch.tensor(np.reshape(np.array(cat_feat1), [1, -1])))
            packed_cat_feat2.append(torch.tensor(np.reshape(np.array(cat_feat2), [1, -1])))
            packed_cat_feat3.append(torch.tensor(np.reshape(np.array(cat_feat3), [1, -1])))
            packed_cat_feat4.append(torch.tensor(np.reshape(np.array(cat_feat4), [1, -1])))

        packed_cat_feat1 = torch.cat(packed_cat_feat1, dim=0)
        packed_cat_feat2 = torch.cat(packed_cat_feat2, dim=0)
        packed_cat_feat3 = torch.cat(packed_cat_feat3, dim=0)
        packed_cat_feat4 = torch.cat(packed_cat_feat4, dim=0)
        b_features = (packed_cat_feat1, packed_cat_feat2, packed_cat_feat3, packed_cat_feat4)
        return b_features

    return collator


def get_collator_train2():
    def collator(data_points):
        packed_cat_feat1 = []
        packed_cat_feat2 = []
        for (cat_feat1, cat_feat2) in data_points:
            packed_cat_feat1.append(torch.tensor(np.reshape(np.array(cat_feat1), [1, -1])))
            packed_cat_feat2.append(torch.tensor(np.reshape(np.array(cat_feat2), [1, -1])))

        packed_cat_feat1 = torch.cat(packed_cat_feat1, dim=0)
        packed_cat_feat2 = torch.cat(packed_cat_feat2, dim=0)
        b_features = (packed_cat_feat1, packed_cat_feat2)
        return b_features

    return collator


def get_collator_val_test():
    def collator(data_points):
        packed_cat_feat = []
        for cat_feat in data_points:
            packed_cat_feat.append(torch.tensor(np.reshape(np.array(cat_feat), [1, -1])))
        packed_cat_feat = torch.cat(packed_cat_feat, dim=0)
        b_features = packed_cat_feat
        return b_features

    return collator

import pdb
import pickle
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MultiheadAttention

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout, with_hidden=False):
        super().__init__()
        layers = []
        self.with_hidden = with_hidden
        prev_layer_size = input_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))
            prev_layer_size = layer_size
        self.layers = nn.Sequential(*layers)
        self.last_layer = nn.Linear(prev_layer_size, output_dim)

    def forward(self, input):
        last_hidden = self.layers(input)
        output = self.last_layer(last_hidden)
        if self.with_hidden:
            return output, last_hidden
        else:
            return output


class BaselineModel(nn.Module):
    def __init__(self, feature_sizes, args):
        super().__init__()
        self.cat_feat_num = len(feature_sizes)

        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, args.embedding_size) for feature_size in feature_sizes])

        self.loss_fnx = CrossEntropyLoss(reduction='none')
        self.loss_fnx_tmp = CrossEntropyLoss()

        self.fm_first_order_embeddings.cuda()
        self.fm_second_order_embeddings.cuda()

        self.bias = torch.nn.Parameter(torch.randn(1))
        self.DNN = nn.ModuleList([
            DNN(args.embedding_size, 1, args.dnn_hidden_units, args.dropout, False),
            DNN(args.embedding_size, 1, args.dnn_hidden_units, args.dropout, False),
        ])
        self.FM_dropout = nn.Dropout(args.dropout)
        for i in range(len(self.DNN)):
            self.DNN[i].cuda()

    def fm_forward(self, cat_feat, task_index):
        feat_index = list(range(self.cat_feat_num))

        # fm part
        fm_first_order_emb_arr = [self.fm_first_order_embeddings[i](cat_feat[:, i]) / 10.0 for i in feat_index]

        fm_second_order_emb_arr = [self.fm_second_order_embeddings[i](cat_feat[:, i]) / 10.0 for i in feat_index]
        fm_second_order_emb = torch.cat(
            [torch.reshape(each, [each.shape[0], 1, each.shape[1]]) for each in fm_second_order_emb_arr], dim=1)

        fm_second_order_emb_sum = torch.sum(fm_second_order_emb, dim=1)
        fm_second_order_emb_sum_square = torch.square(fm_second_order_emb_sum)

        fm_second_order_emb_square = torch.square(fm_second_order_emb)
        fm_second_order_emb_square_sum = torch.sum(fm_second_order_emb_square, dim=1)

        fm_second_order = (fm_second_order_emb_sum_square - fm_second_order_emb_square_sum) * 0.5

        fm_second_order = self.FM_dropout(fm_second_order)
        deep_out = self.DNN[task_index](fm_second_order)
        total_sum = torch.reshape(deep_out, [-1]) + torch.sum(fm_second_order, 1) + self.bias
        total_sum = torch.reshape(total_sum, [-1, 1])

        return total_sum

    def forward(self, input, flag):
        if flag == 1:
            cat_feat1, cat_feat2, cat_feat3, cat_feat4 = input
            score1_task1 = self.fm_forward(cat_feat1, task_index=0)
            score2_task1 = self.fm_forward(cat_feat2, task_index=0)
            score1_task2 = self.fm_forward(cat_feat3, task_index=1)
            score2_task2 = self.fm_forward(cat_feat4, task_index=1)

            return score1_task1, score2_task1, score1_task2, score2_task2

        elif flag == 2:
            cat_feat1, cat_feat2 = input
            score1 = self.fm_forward(cat_feat1, task_index=0)
            score2 = self.fm_forward(cat_feat2, task_index=0)

            return score1, score2

        elif flag == 0:
            cat_feat = input
            score = self.fm_forward(cat_feat, task_index=0)

            return score

        else:
            raise ValueError


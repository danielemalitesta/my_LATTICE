import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pandas as pd
from torch_sparse import SparseTensor, mul, sum, fill_diag, matmul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.parser import parse_args
from Models import LATTICE
from utility.batch_test import *

args = parse_args()


class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.model_name = args.model_name
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().cuda()

        image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))

        if args.missing_features:
            self.image_feat_dim = image_feats.shape[-1]
            self.text_feat_dim = text_feats.shape[-1]

            self.ui_graph = data_config['R']

            if args.masked_items_image:
                masked_items_image = pd.read_csv(args.masked_items_image, sep='\t', header=None)[0].tolist()
            else:
                raise FileNotFoundError('File for masked items not found!')

            if args.masked_items_text:
                masked_items_text = pd.read_csv(args.masked_items_text, sep='\t', header=None)[0].tolist()
            else:
                raise FileNotFoundError('File for masked items not found!')

            if args.strategy == 'zeros':
                image_feats[masked_items_image] = np.zeros((1, self.image_feat_dim))
                text_feats[masked_items_text] = np.zeros((1, self.text_feat_dim))
            elif args.strategy == 'mean':
                mask = np.ones(self.n_items, dtype=bool)
                mask[masked_items_image] = False
                result = image_feats[mask]
                mean_ = result.mean(axis=0)
                image_feats[masked_items_image] = mean_

                mask = np.ones(self.n_items, dtype=bool)
                mask[masked_items_text] = False
                result = text_feats[mask]
                mean_ = result.mean(axis=0)
                text_feats[masked_items_text] = mean_
            elif args.strategy == 'random':
                image_feats[masked_items_image] = np.random.rand(len(masked_items_image), self.image_feat_dim)
                text_feats[masked_items_text] = np.random.rand(len(masked_items_text), self.text_feat_dim)
            elif args.strategy == 'feat_prop':
                if args.feat_prop == 'co':
                    item_item = self.ui_graph.transpose().dot(self.ui_graph).toarray()
                    # get non masked items
                    non_masked_items = list(set(list(range(self.n_items))).difference(masked_items_image))
                    # binarize adjacency matrix
                    item_item[item_item >= 1] = 1.0
                    # set zeros as initialization
                    image_feats[masked_items_image] = np.zeros((1, self.image_feat_dim))
                    # get sparse adjacency matrix
                    row, col = item_item.nonzero()
                    edge_index = np.array([row, col])
                    edge_index = torch.tensor(edge_index, dtype=torch.int64)
                    adj = SparseTensor(row=edge_index[0],
                                       col=edge_index[1],
                                       sparse_sizes=(self.n_items, self.n_items))
                    # normalize adjacency matrix
                    adj = self.compute_normalized_laplacian(adj, 0.5)
                    # feature propagation
                    propagated_features = torch.tensor(image_feats)
                    for _ in range(args.prop_layers):
                        propagated_features = matmul(adj, propagated_features)
                        propagated_features[non_masked_items] = torch.tensor(image_feats[non_masked_items])
                    image_feats[masked_items_image] = propagated_features[masked_items_image].detach().cpu().numpy()

                    item_item = self.ui_graph.transpose().dot(self.ui_graph).toarray()
                    # get non masked items
                    non_masked_items = list(set(list(range(self.n_items))).difference(masked_items_text))
                    # binarize adjacency matrix
                    item_item[item_item >= 1] = 1.0
                    # set zeros as initialization
                    text_feats[masked_items_text] = np.zeros((1, self.text_feat_dim))
                    # get sparse adjacency matrix
                    row, col = item_item.nonzero()
                    edge_index = np.array([row, col])
                    edge_index = torch.tensor(edge_index, dtype=torch.int64)
                    adj = SparseTensor(row=edge_index[0],
                                       col=edge_index[1],
                                       sparse_sizes=(self.n_items, self.n_items))
                    # normalize adjacency matrix
                    adj = self.compute_normalized_laplacian(adj, 0.5)
                    # feature propagation
                    propagated_features = torch.tensor(text_feats)
                    for _ in range(args.prop_layers):
                        propagated_features = matmul(adj, propagated_features)
                        propagated_features[non_masked_items] = torch.tensor(text_feats[non_masked_items])
                    text_feats[masked_items_text] = propagated_features[masked_items_text].detach().cpu().numpy()
                elif args.feat_prop == 'rev':
                    pass
                else:
                    raise NotImplementedError('This aggregation has not been implemented yet!')
            else:
                raise NotImplementedError('This strategy has not been implemented yet!')

        self.model = LATTICE(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats,
                             text_feats)
        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    @staticmethod
    def compute_normalized_laplacian(adj, norm):
        adj = fill_diag(adj, 0.)
        deg = sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow_(-norm)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=True)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            build_item_graph = True
            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1
                ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=build_item_graph)
                build_item_graph = False
                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward(retain_graph=True)
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                epoch, time() - t1, loss, mf_loss, emb_loss)
            training_time_list.append(time() - t1)
            print(perf_str)

            if epoch % args.verbose != 0:
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)

            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])
            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]:  val==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                print(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                           'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, test_ret['recall'][0],
                            test_ret['recall'][-1],
                            test_ret['precision'][0], test_ret['precision'][-1], test_ret['hit_ratio'][0],
                            test_ret['hit_ratio'][-1],
                            test_ret['ndcg'][0], test_ret['ndcg'][-1])
                print(perf_str)
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        print(test_ret)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu


if __name__ == '__main__':

    if args.missing_features == 'True':
        args.missing_features = True
    else:
        args.missing_features = False

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    R = data_generator.get_R()
    config['norm_adj'] = norm_adj
    config['R'] = R

    trainer = Trainer(data_config=config)
    trainer.train()


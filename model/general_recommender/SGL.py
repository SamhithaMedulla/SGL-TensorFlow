"""
Paper: Self-supervised Graph Learning for Recommendation
Author: Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian & Xing Xie
Reference: https://github.com/hexiangnan/LightGCN
"""

import os
import sys
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, lil_matrix
from model.AbstractRecommender import AbstractRecommender
from util import timer, tool, learner
from util import l2_loss, inner_product, log_loss
from data import PairwiseSampler, PairwiseSamplerV2, PointwiseSamplerV2
from util.cython.random_choice import randint_choice
from util.tool import randint_choice as randint_choice_v2
from time import time
from collections.abc import Iterable


class SGL(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SGL, self).__init__(dataset, conf)

        self.model_name = conf["recommender"]
        self.conf = conf
        self.dataset_name = conf["data.input.dataset"]
        self.lr = conf["lr"]
        self.reg = conf["reg"]
        self.embedding_size = conf["embed_size"]
        self.learner = conf["learner"]
        self.batch_size = conf["batch_size"]
        self.test_batch_size = conf["test_batch_size"]
        self.epochs = conf["epochs"]
        self.verbose = conf["verbose"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.n_layers = conf["n_layers"]
        self.adj_type = conf["adj_type"]
        self.stop_cnt = conf["stop_cnt"]

        self.aug_type = conf["aug_type"]
        self.ssl_mode = conf["ssl_mode"]
        self.ssl_ratio = conf["ssl_ratio"]
        self.ssl_temp = conf["ssl_temp"]
        self.ssl_reg = conf["ssl_reg"]

        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.all_users = list(self.user_pos_train.keys())

        self.training_user, self.training_item = self._get_training_data()
        self.norm_adj = self.create_adj_mat(is_subgraph=False)
        self.sub_mat = {}  # empty dictionary for sparse inputs
        self.best_result = np.full(5, -np.inf, dtype=float)
        self.best_epoch = 0
        self.sess = sess
        self.model_str = "#layers=%d-%s-reg%.0e" % (self.n_layers, self.adj_type, self.reg)
        self.model_str += "/ratio=%.1f-mode=%s-temp=%.2f-reg=%.0e" % (self.ssl_ratio, self.ssl_mode, self.ssl_temp, self.ssl_reg)
        self.pretrain = conf["pretrain"]
        if self.pretrain:
            self.epochs = 0
        self.save_flag = conf["save_flag"]
        if self.pretrain or self.save_flag:
            self.tmp_model_folder = conf["proj_path"] + "model_tmp/%s/%s/%s/" % (self.dataset_name, self.model_name, self.model_str)
            self.save_folder = conf["proj_path"] + "dataset/pretrain-embeddings-%s/%s/n_layers=%d/" % (self.dataset_name, self.model_name, self.n_layers)
            tool.ensureDir(self.tmp_model_folder)
            tool.ensureDir(self.save_folder)
            # initial sub_mat entries (may be overwritten later)
            self.sub_mat = {
                "adj_indices_sub1": np.zeros((self.n_users, self.n_items)),
                "adj_indices_sub2": np.zeros((self.n_users, self.n_items)),
            }

    def _get_training_data(self):
        user_list, item_list = self.dataset.get_train_interactions()
        return user_list, item_list

    def create_adj_mat(self, is_subgraph=False, aug_type=0):
        n_nodes = self.n_users + self.n_items
        if is_subgraph and aug_type in [0, 1, 2] and self.ssl_ratio > 0:
            if aug_type == 0:
                drop_user_idx = randint_choice(self.n_users, size=int(self.n_users * self.ssl_ratio), replace=False)
                drop_item_idx = randint_choice(self.n_items, size=int(self.n_items * self.ssl_ratio), replace=False)
                indicator_user = np.ones(self.n_users, dtype=np.float32)
                indicator_item = np.ones(self.n_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.
                indicator_item[drop_item_idx] = 0.
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                R = sp.csr_matrix(
                    (np.ones_like(self.training_user, dtype=np.float32),
                     (self.training_user, self.training_item)),
                    shape=(self.n_users, self.n_items))
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                (user_np_keep, item_np_keep) = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.n_users)),
                                        shape=(n_nodes, n_nodes))
            elif aug_type in [1, 2]:
                keep_idx = randint_choice(len(self.training_user),
                                          size=int(len(self.training_user) * (1 - self.ssl_ratio)),
                                          replace=False)
                user_np = np.array(self.training_user)[keep_idx]
                item_np = np.array(self.training_item)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        else:
            user_np = np.array(self.training_user)
            item_np = np.array(self.training_item)
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # Avoid divide-by-zero for normalization.
        rowsum = np.array(adj_mat.sum(1)).flatten()
        d_inv = np.power(rowsum, -0.5, where=(rowsum != 0))
        d_inv[rowsum == 0] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def _create_variable(self):
        with tf.name_scope("input_data"):
            # Training data placeholders.
            self.input_data = tf.compat.v1.placeholder(tf.int32, shape=[None, 2], name="input_data")
            self.users = tf.cast(self.input_data[:, 0], tf.int32)
            self.pos_items = tf.cast(self.input_data[:, 1], tf.int32)
            self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_items")

            # Sparse tensor placeholders.
            self.adj_indices_sub1 = tf.compat.v1.placeholder(tf.int64, shape=[None, 2], name="adj_indices_sub1")
            self.adj_values_sub1 = tf.compat.v1.placeholder(tf.float32, shape=[None], name="adj_values_sub1")
            self.adj_shape_sub1 = tf.compat.v1.placeholder(tf.int64, shape=[2], name="adj_shape_sub1")
            self.adj_indices_sub2 = tf.compat.v1.placeholder(tf.int64, shape=[None, 2], name="adj_indices_sub2")
            self.adj_values_sub2 = tf.compat.v1.placeholder(tf.float32, shape=[None], name="adj_values_sub2")
            self.adj_shape_sub2 = tf.compat.v1.placeholder(tf.int64, shape=[2], name="adj_shape_sub2")

            # Build SparseTensors from placeholders.
            self.sub_mat["sub_mat_sub1"] = tf.SparseTensor(self.adj_indices_sub1,
                                                           self.adj_values_sub1,
                                                           self.adj_shape_sub1)
            self.sub_mat["sub_mat_sub2"] = tf.SparseTensor(self.adj_indices_sub2,
                                                           self.adj_values_sub2,
                                                           self.adj_shape_sub2)
            # Also keep placeholders for feeding.
            self.sub_mat.update({
                "adj_indices_sub1": self.adj_indices_sub1,
                "adj_values_sub1": self.adj_values_sub1,
                "adj_shape_sub1": self.adj_shape_sub1,
                "adj_indices_sub2": self.adj_indices_sub2,
                "adj_values_sub2": self.adj_values_sub2,
                "adj_shape_sub2": self.adj_shape_sub2,
            })
            if self.aug_type not in [0, 1]:
                for k in range(1, self.n_layers + 1):
                    key1 = "sub_mat_sub1%d" % k
                    key2 = "sub_mat_sub2%d" % k
                    self.sub_mat[key1] = tf.compat.v1.sparse_placeholder(tf.float32,
                                                                          shape=[self.n_users + self.n_items,
                                                                                 self.n_users + self.n_items],
                                                                          name=key1)
                    self.sub_mat[key2] = tf.compat.v1.sparse_placeholder(tf.float32,
                                                                          shape=[self.n_users + self.n_items,
                                                                                 self.n_users + self.n_items],
                                                                          name=key2)
                    self.sub_mat["adj_indices_sub1%d" % k] = tf.compat.v1.placeholder(tf.int64,
                                                                                       name="adj_indices_sub1%d" % k)
                    self.sub_mat["adj_values_sub1%d" % k] = tf.compat.v1.placeholder(tf.float32,
                                                                                    name="adj_values_sub1%d" % k)
                    self.sub_mat["adj_shape_sub1%d" % k] = tf.compat.v1.placeholder(tf.int64,
                                                                                     name="adj_shape_sub1%d" % k)
                    self.sub_mat["adj_indices_sub2%d" % k] = tf.compat.v1.placeholder(tf.int64,
                                                                                       name="adj_indices_sub2%d" % k)
                    self.sub_mat["adj_values_sub2%d" % k] = tf.compat.v1.placeholder(tf.float32,
                                                                                    name="adj_values_sub2%d" % k)
                    self.sub_mat["adj_shape_sub2%d" % k] = tf.compat.v1.placeholder(tf.int64,
                                                                                     name="adj_shape_sub2%d" % k)
        with tf.name_scope("embedding_init"):
            self.weights = dict()
            initializer = tf.keras.initializers.GlorotUniform()
            if self.pretrain:
                pretrain_user_embedding = np.load(self.save_folder + "user_embeddings.npy")
                pretrain_item_embedding = np.load(self.save_folder + "item_embeddings.npy")
                self.weights["user_embedding"] = tf.Variable(pretrain_user_embedding,
                                                             name="user_embedding", dtype=tf.float32)
                self.weights["item_embedding"] = tf.Variable(pretrain_item_embedding,
                                                             name="item_embedding", dtype=tf.float32)
            else:
                self.weights["user_embedding"] = tf.Variable(initializer([self.n_users, self.embedding_size]),
                                                             name="user_embedding")
                self.weights["item_embedding"] = tf.Variable(initializer([self.n_items, self.embedding_size]),
                                                             name="item_embedding")

    def build_graph(self):
        self._create_variable()
        with tf.name_scope("inference"):
            self.ua_embeddings, self.ia_embeddings, \
            self.ua_embeddings_sub1, self.ia_embeddings_sub1, \
            self.ua_embeddings_sub2, self.ia_embeddings_sub2 = self._create_lightgcn_SSL_embed()

        # Load training data.
        train_data_path = r"C:\Users\samhi\Documents\SGL-TensorFlow\dataset\amazon-book.train"
        train_data = pd.read_csv(train_data_path, header=None, names=["user_id", "item_id", "rating"])
        train_input_data = train_data[["user_id", "item_id"]].values

        with tf.name_scope("loss"):
            if self.pretrain:
                self.ssl_loss = tf.constant(0, dtype=tf.float32)
            else:
                if self.ssl_mode in ["user_side", "item_side", "both_side"]:
                    self.ssl_loss = self.calc_ssl_loss_v2()
                elif self.ssl_mode in ["merge"]:
                    self.ssl_loss = self.calc_ssl_loss_v3()
                else:
                    raise ValueError("Invalid ssl_mode!")
        self.sl_loss, self.emb_loss = self.create_bpr_loss()
        self.loss = self.sl_loss + self.emb_loss + self.ssl_loss

        with tf.name_scope("learner"):
            # Use a TF v1 optimizer minimizing the loss.
            if self.learner.lower() == "adam":
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
            elif self.learner.lower() == "sgd":
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise ValueError("Unsupported learner: " + self.learner)
            self.opt = optimizer.minimize(self.loss)
        self.saver = tf.compat.v1.train.Saver()

    def _create_lightgcn_SSL_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        ego_embeddings = tf.concat([self.weights["user_embedding"], self.weights["item_embedding"]], axis=0)
        ego_embeddings_sub1 = ego_embeddings
        ego_embeddings_sub2 = ego_embeddings
        all_embeddings = [ego_embeddings]
        all_embeddings_sub1 = [ego_embeddings_sub1]
        all_embeddings_sub2 = [ego_embeddings_sub2]

        for k in range(1, self.n_layers + 1):
            ego_embeddings = tf.sparse.sparse_dense_matmul(adj_mat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            if self.aug_type in [0, 1]:
                ego_embeddings_sub1 = tf.sparse.sparse_dense_matmul(self.sub_mat["sub_mat_sub1"], ego_embeddings_sub1)
                ego_embeddings_sub2 = tf.sparse.sparse_dense_matmul(self.sub_mat["sub_mat_sub2"], ego_embeddings_sub2)
            else:
                ego_embeddings_sub1 = tf.sparse.sparse_dense_matmul(self.sub_mat["sub_mat_sub1%d" % k], ego_embeddings_sub1)
                ego_embeddings_sub2 = tf.sparse.sparse_dense_matmul(self.sub_mat["sub_mat_sub2%d" % k], ego_embeddings_sub2)
            all_embeddings_sub1.append(ego_embeddings_sub1)
            all_embeddings_sub2.append(ego_embeddings_sub2)

        all_embeddings = tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1)
        all_embeddings_sub1 = tf.reduce_mean(tf.stack(all_embeddings_sub1, axis=1), axis=1)
        all_embeddings_sub2 = tf.reduce_mean(tf.stack(all_embeddings_sub2, axis=1), axis=1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], axis=0)
        u_g_embeddings_sub1, i_g_embeddings_sub1 = tf.split(all_embeddings_sub1, [self.n_users, self.n_items], axis=0)
        u_g_embeddings_sub2, i_g_embeddings_sub2 = tf.split(all_embeddings_sub2, [self.n_users, self.n_items], axis=0)
        return u_g_embeddings, i_g_embeddings, u_g_embeddings_sub1, i_g_embeddings_sub1, u_g_embeddings_sub2, i_g_embeddings_sub2

    def calc_ssl_loss_v2(self):
        if self.ssl_mode in ["user_side", "both_side"]:
            user_emb1 = tf.gather(self.ua_embeddings_sub1, self.users)
            if len(user_emb1.shape) == 1:
                user_emb1 = tf.reshape(user_emb1, [1, -1])
            user_emb2 = tf.nn.embedding_lookup(self.ua_embeddings_sub2, self.users)
            if len(user_emb2.shape) == 1:
                user_emb2 = tf.reshape(user_emb2, [1, -1])
            normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, axis=1)
            normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, axis=1)
            normalize_all_user_emb2 = tf.nn.l2_normalize(self.ua_embeddings_sub2, axis=1)
            pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
            ttl_score_user = tf.reduce_sum(tf.exp(tf.matmul(normalize_user_emb1, normalize_all_user_emb2, transpose_b=True) / self.ssl_temp), axis=1)
            pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
            ssl_loss_user = -tf.reduce_sum(tf.math.log(pos_score_user / ttl_score_user))
        if self.ssl_mode in ["item_side", "both_side"]:
            item_emb1 = tf.nn.embedding_lookup(self.ia_embeddings_sub1, self.pos_items)
            item_emb2 = tf.nn.embedding_lookup(self.ia_embeddings_sub2, self.pos_items)
            if len(item_emb1.shape) == 1:
                item_emb1 = tf.reshape(item_emb1, [1, -1])
            if len(item_emb2.shape) == 1:
                item_emb2 = tf.reshape(item_emb2, [1, -1])
            normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, axis=1)
            normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, axis=1)
            normalize_all_item_emb2 = tf.nn.l2_normalize(self.ia_embeddings_sub2, axis=1)
            pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
            ttl_score_item = tf.reduce_sum(tf.exp(tf.matmul(normalize_item_emb1, normalize_all_item_emb2, transpose_b=True) / self.ssl_temp), axis=1)
            pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
            ssl_loss_item = -tf.reduce_sum(tf.math.log(pos_score_item / ttl_score_item))
        if self.ssl_mode == "user_side":
            ssl_loss = self.ssl_reg * ssl_loss_user
        elif self.ssl_mode == "item_side":
            ssl_loss = self.ssl_reg * ssl_loss_item
        else:
            ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss

    def calc_ssl_loss(self):
        user_emb1 = tf.gather(self.ua_embeddings_sub1, tf.convert_to_tensor(self.users))
        user_emb2 = tf.nn.embedding_lookup(self.ua_embeddings_sub2, self.users)
        normalize_user_emb1 = tf.nn.l2_normalize(user_emb1, 1)
        normalize_user_emb2 = tf.nn.l2_normalize(user_emb2, 1)
        item_emb1 = tf.nn.embedding_lookup(self.ia_embeddings_sub1, self.pos_items)
        item_emb2 = tf.nn.embedding_lookup(self.ia_embeddings_sub2, self.pos_items)
        normalize_item_emb1 = tf.nn.l2_normalize(item_emb1, 1)
        normalize_item_emb2 = tf.nn.l2_normalize(item_emb2, 1)
        pos_score_user = tf.reduce_sum(tf.multiply(normalize_user_emb1, normalize_user_emb2), axis=1)
        ttl_score_user = tf.reduce_sum(tf.exp(tf.matmul(normalize_user_emb1, normalize_user_emb2, transpose_b=True) / self.ssl_temp), axis=1)
        pos_score_item = tf.reduce_sum(tf.multiply(normalize_item_emb1, normalize_item_emb2), axis=1)
        ttl_score_item = tf.reduce_sum(tf.exp(tf.matmul(normalize_item_emb1, normalize_item_emb2, transpose_b=True) / self.ssl_temp), axis=1)
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ssl_loss_user = -tf.reduce_sum(tf.log(pos_score_user / ttl_score_user))
        ssl_loss_item = -tf.reduce_sum(tf.log(pos_score_item / ttl_score_item))
        ssl_loss = self.ssl_reg * (ssl_loss_user + ssl_loss_item)
        return ssl_loss

    def calc_ssl_loss_v3(self):
        batch_users, _ = tf.unique(self.users)
        user_emb1 = tf.nn.embedding_lookup(self.ua_embeddings_sub1, batch_users)
        user_emb2 = tf.nn.embedding_lookup(self.ua_embeddings_sub2, batch_users)
        batch_items, _ = tf.unique(self.pos_items)
        item_emb1 = tf.nn.embedding_lookup(self.ia_embeddings_sub1, batch_items)
        item_emb2 = tf.nn.embedding_lookup(self.ia_embeddings_sub2, batch_items)
        emb_merge1 = tf.concat([user_emb1, item_emb1], axis=0)
        emb_merge2 = tf.concat([user_emb2, item_emb2], axis=0)
        normalize_emb_merge1 = tf.nn.l2_normalize(emb_merge1, 1)
        normalize_emb_merge2 = tf.nn.l2_normalize(emb_merge2, 1)
        pos_score = tf.reduce_sum(tf.multiply(normalize_emb_merge1, normalize_emb_merge2), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(tf.matmul(normalize_emb_merge1, normalize_emb_merge2, transpose_b=True) / self.ssl_temp), axis=1)
        pos_score = tf.exp(pos_score / self.ssl_temp)
        ssl_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score))
        ssl_loss = self.ssl_reg * ssl_loss
        return ssl_loss

    def create_bpr_loss(self):
        batch_u_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        batch_pos_i_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        batch_neg_i_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        batch_u_embeddings_pre = tf.nn.embedding_lookup(self.weights["user_embedding"], self.users)
        batch_pos_i_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.pos_items)
        batch_neg_i_embeddings_pre = tf.nn.embedding_lookup(self.weights["item_embedding"], self.neg_items)
        regularizer = l2_loss(batch_u_embeddings_pre, batch_pos_i_embeddings_pre, batch_neg_i_embeddings_pre)
        emb_loss = self.reg * regularizer
        pos_scores = inner_product(batch_u_embeddings, batch_pos_i_embeddings)
        neg_scores = inner_product(batch_u_embeddings, batch_neg_i_embeddings)
        bpr_loss = tf.reduce_sum(log_loss(pos_scores - neg_scores))
        self.grad_score = 1 - tf.sigmoid(pos_scores - neg_scores)
        if len(batch_u_embeddings.shape) == 1:
            self.grad_user_embed = self.grad_score * tf.sqrt(tf.reduce_sum(tf.square(batch_u_embeddings)))
        else:
            self.grad_user_embed = self.grad_score * tf.sqrt(tf.reduce_sum(tf.square(batch_u_embeddings), axis=1))
        if len(batch_pos_i_embeddings.shape) == 1:
            self.grad_item_embed = self.grad_score * tf.sqrt(tf.reduce_sum(tf.square(batch_pos_i_embeddings)))
        else:
            self.grad_item_embed = self.grad_score * tf.sqrt(tf.reduce_sum(tf.square(batch_pos_i_embeddings), axis=1))
        if len(batch_u_embeddings.shape) == 1:
            emb_loss += tf.reduce_sum(tf.square(batch_u_embeddings))
        else:
            emb_loss += tf.reduce_sum(tf.square(batch_u_embeddings), axis=1)
        return bpr_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.array(np.mat([coo.row, coo.col]).transpose())
        return indices, coo.data, coo.shape

    def train_model(self):
        data_iter = PairwiseSamplerV2(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True)
        self.logger.info(self.evaluator.metrics_info())
        buf, _ = self.evaluate()
        self.logger.info("\t\t%s" % buf)
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            sub_mat = {}
            if self.aug_type in [0, 1]:
                indices1, values1, shape1 = self._convert_csr_to_sparse_tensor_inputs(
                    self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                indices2, values2, shape2 = self._convert_csr_to_sparse_tensor_inputs(
                    self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                sub_mat["indices_sub1"] = indices1
                sub_mat["values_sub1"] = values1
                sub_mat["shape_sub1"] = shape1
                sub_mat["indices_sub2"] = indices2
                sub_mat["values_sub2"] = values2
                sub_mat["shape_sub2"] = shape2
            else:
                for k in range(1, self.n_layers + 1):
                    indices1, values1, shape1 = self._convert_csr_to_sparse_tensor_inputs(
                        self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                    indices2, values2, shape2 = self._convert_csr_to_sparse_tensor_inputs(
                        self.create_adj_mat(is_subgraph=True, aug_type=self.aug_type))
                    sub_mat["indices_sub1%d" % k] = indices1
                    sub_mat["values_sub1%d" % k] = values1
                    sub_mat["shape_sub1%d" % k] = shape1
                    sub_mat["indices_sub2%d" % k] = indices2
                    sub_mat["values_sub2%d" % k] = values2
                    sub_mat["shape_sub2%d" % k] = shape2

            total_loss, total_ssl_loss, total_emb_loss = 0.0, 0.0, 0.0
            training_start_time = time()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed_dict = {
                    self.users: bat_users,
                    self.pos_items: bat_pos_items,
                    self.neg_items: bat_neg_items,
                }
                if self.aug_type in [0, 1]:
                    feed_dict.update({
                        self.sub_mat["adj_indices_sub1"]: sub_mat["indices_sub1"],
                        self.sub_mat["adj_values_sub1"]: sub_mat["values_sub1"],
                        self.sub_mat["adj_shape_sub1"]: sub_mat["shape_sub1"],
                        self.sub_mat["adj_indices_sub2"]: sub_mat["indices_sub2"],
                        self.sub_mat["adj_values_sub2"]: sub_mat["values_sub2"],
                        self.sub_mat["adj_shape_sub2"]: sub_mat["shape_sub2"],
                    })
                else:
                    for k in range(1, self.n_layers + 1):
                        feed_dict.update({
                            self.sub_mat["adj_indices_sub1%d" % k]: sub_mat["indices_sub1%d" % k],
                            self.sub_mat["adj_values_sub1%d" % k]: sub_mat["values_sub1%d" % k],
                            self.sub_mat["adj_shape_sub1%d" % k]: sub_mat["shape_sub1%d" % k],
                            self.sub_mat["adj_indices_sub2%d" % k]: sub_mat["indices_sub2%d" % k],
                            self.sub_mat["adj_values_sub2%d" % k]: sub_mat["values_sub2%d" % k],
                            self.sub_mat["adj_shape_sub2%d" % k]: sub_mat["shape_sub2%d" % k],
                        })
                loss, ssl_loss, emb_loss, _ = self.sess.run((self.loss, self.ssl_loss, self.emb_loss, self.opt),
                                                            feed_dict=feed_dict)
                total_loss += loss
                total_ssl_loss += ssl_loss
                total_emb_loss += emb_loss

            if np.isnan(total_loss):
                self.logger.info("Nan is encountered!")
                sys.exit(1)
            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f + %.4f, time: %f]" %
                             (epoch,
                              total_loss / data_iter.num_trainings,
                              (total_loss - total_ssl_loss - total_emb_loss) / data_iter.num_trainings,
                              total_ssl_loss / data_iter.num_trainings,
                              total_emb_loss / data_iter.num_trainings,
                              time() - training_start_time))
            if epoch % self.verbose == 0 and epoch > self.conf["start_testing_epoch"]:
                buf, flag = self.evaluate()
                self.logger.info("epoch %d:\t%s" % (epoch, buf))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        self.saver.save(self.sess, self.tmp_model_folder)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info("Loading from the saved model.")
            self.saver.restore(self.sess, self.tmp_model_folder)
            uebd, iebd = self.sess.run([self.weights["user_embedding"], self.weights["item_embedding"]])
            np.save(self.save_folder + "user_embeddings.npy", uebd)
            np.save(self.save_folder + "item_embeddings.npy", iebd)
            buf, _ = self.evaluate()
        elif self.pretrain:
            buf, _ = self.evaluate()
        else:
            buf = "\t".join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)
        if hasattr(self.logger, "handlers"):
            for handler in self.logger.handlers:
                handler.flush()
                
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run(
            [self.ua_embeddings, self.ia_embeddings])
        flag = False
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        self.logger.info("\tEvaluation result: %s" % buf)
        if hasattr(self.logger, "handlers"):
            for handler in self.logger.handlers:
                handler.flush()
        return buf, flag
    
    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            user_embed = self._cur_user_embeddings[user_ids]
            items_embed = self._cur_item_embeddings[candidate_items]
            ratings = np.sum(np.multiply(user_embed, items_embed), 1)
        return ratings

    
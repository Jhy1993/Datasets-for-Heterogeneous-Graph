#!/usr/bin/env python

from time import time
import scipy.sparse as sp
import random
import numpy as np
import os
import scipy.io as sio


"""
抽取 U-M
meta-path
U: UU.mat, UMU.mat, 943
I: MM.mat, MGM.mat, 1682

"""


def load_data(fp):
    # note that ID start from 1 in data, so true ID = id - 1
    # user_user(knn).dat  中边是无向的 且存储了两次
    neighbor_list = []
    num_A = 0
    num_B = 0
    num_AB = 0
    edge = []

    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            line = [int(i)-1 for i in line[:2]]
            num_A = max(line[0], num_A)
            num_B = max(line[1], num_B)
            num_AB += 1
            edge.append([line[0], line[1]])
    real_num_A = num_A + 1
    real_num_B = num_B + 1
    adj_mat = sp.dok_matrix((real_num_A, real_num_B), dtype=np.float32)
    for ed in edge:
        adj_mat[ed[0], ed[1]] = 1.0
        adj_mat[ed[1], ed[0]] = 1.0

    print('load: {}, A: {}, B: {}, A-B: {}'.format(fp, num_A, num_B, num_AB))
    print('adj mat sum: {}'.format(np.sum(adj_mat)))
    return edge


# load_data('./user_user(knn).dat')
# load_data('./user_movie.dat')
def check_data(fp='./user_movie.dat', modify=False):
    user_list = []
    movie_list = []
    k = 0
    edge_list = []
    if modify:
        new_user_list = []
        new_movie_list = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            # print(line)
            user_list.append(int(line[0]))
            movie_list.append(int(line[1]))
            k += 1
            if modify:
                edge_list.append(line[0] + '_'+line[1])
                edge_list.append(line[1] + '_' + line[0])
                new_user_list.append(int(line[0]))
                new_movie_list.append(int(line[1]))

    user_set = set(user_list)
    movie_set = set(movie_list)
    print('before modify')
    print(len(user_set), len(movie_set), k)
    print(min(user_list), max(user_list))
    print(min(movie_list), max(movie_list))
    if modify:
        print('modify data....')
        edge_set = set(edge_list)
        print(len(edge_set), min(new_user_list), max(new_user_list),
              min(new_movie_list), max(new_movie_list))
        file = open('./jhy_movielens/' + fp, 'w')
        row_list = []
        col_list = []
        data_list = []
        for ed in edge_set:
            ed = ed.split('_')
            file.write(ed[0])
            file.write('\t')
            file.write(ed[1])
            file.write('\n')
            row_list.append(int(ed[0])-1)
            col_list.append(int(ed[1])-1)
            data_list.append(1.0)
        row = np.array(row_list)
        col = np.array(col_list)
        data = np.array(data_list)

        mat = sp.coo_matrix((data, (row, col)),
                            shape=(max(new_user_list), max(new_movie_list)))
        print(mat.shape, np.sum(mat), np.sum(
            mat)/(mat.shape[0] * mat.shape[1]))
        sio.savemat('./jhy_movielens/' + fp, {'UU': mat})
        # ========  save mat ======


# 先检查，在modify
# check_data('user_user(knn).dat',
#               modify=True
#            )


def build_MM(fp):
    # 现有的movie_movie(knn).dat中只有1657个moive，少于实际的1682，
    # 为了保证所有movie， 所以决定抽新的， MM的相似性用其genre vector的相似性，
    # 然后每个movie取Top-5，然后双向化(m2的top-5里面有m1,但是m1的top-5里可能没有m2)

    #  如果用feature lookup，就不用保证所有的movie了？
    genre_mat = np.zeros((1682, 18))
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            m1 = int(line[0]) - 1
            m2 = int(line[1]) - 1
            genre_mat[m1, m2] = 1.0
    print(genre_mat.shape, np.sum(genre_mat), np.sum(genre_mat) /
          (genre_mat.shape[0] * genre_mat.shape[1]))
    # genre_mat_sp = genre_mat.tocsr()
    sio.savemat('./jhy_movielens/MG.mat', {'MG': genre_mat})
    MGM_tmp = np.dot(genre_mat, genre_mat.transpose())
    MGM = np.where(MGM_tmp > 0, 1.0, 0.0)
    print(MGM.shape, np.sum(MGM), np.sum(MGM)/(MGM.shape[0] * MGM.shape[1]))
    sio.savemat('./jhy_movielens/MGM.mat', {'MGM': MGM})


# build_MM('movie_genre.dat')


def split_rating(fp, sp=0.8):
    # 划分sp作为训练，
    # 注意 需要保证所有um在训练集中都出现
    # 可以试着每个user都取前80%的交互作为训练
    u_neigh_list = []
    u = []
    m = []
    num_u = 0
    num_m = 0
    num_rating = 0
    rate_list = []
    with open(fp, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            l1 = int(line[0]) - 1
            l2 = int(line[1]) - 1
            num_u = max(num_u, l1)
            num_m = max(num_m, l2)
            num_rating += 1
            rate_list.append([l1, l2])
    print(f"u: {num_u+1}, m: {num_m+1}, r: {num_rating}")
    random.shuffle(rate_list)
    sp = int(len(rate_list) * sp)
    train_rate_list = rate_list[:sp]
    print(f"train inst num: {len(train_rate_list)}")
    e1_list = []
    e2_list = []
    for e in train_rate_list:
        e1_list.append(e[0])
        e2_list.append(e[1])
    print(f"e1: {len(set(e1_list))}, e2: {len(set(e2_list))}")
        
        
split_rating('user_movie.dat')


class Data(object):
    def __init__(self, path, batch_size):
        """

        Args:
            path:
            batch_size:
        """
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}
        # only user in train
        self.exist_users = []

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        # 因为下标从0开始
        self.n_items += 1
        self.n_users += 1

        self.print_statistics()
        # 不是评分矩阵，是交互矩阵，这里是隐式的交互
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        # TODO 这里合在一起写， 和分别写with 在read有啥区别？
        with open(train_file) as f_train:
            with open(test_file) as f_test:
                for l in f_train.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, train_items = items[0], items[1:]
                    # 为什么只对训练集的数据填充到R矩阵中?
                    # 因为后面只通过训练集的边进行聚合
                    # 那么训练集包含所有的u i吗？
                    for i in train_items:
                        self.R[uid, i] = 1.
                        # self.R[uid][i] = 1

                    self.train_items[uid] = train_items

                for l in f_test.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip('\n')
                    try:
                        items = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_items = items[0], items[1:]
                    # TODO 都是存储 user-item字典，为什么训练的叫train_items，测试的叫test_set、、、
                    self.test_set[uid] = test_items

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        # 创建这个矩阵是为了使用GCN
        # 这里的A是同时包含U和I的邻接情况
        # 但是边的情况，只根据train中来建立，也就是R
        adj_mat = sp.dok_matrix((self.n_users + self.n_items,
                                 self.n_users + self.n_items),
                                dtype=np.float32)
        # 转化为邻接的link list
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        # 这里对[U+I, U+I]的矩阵补上，左上角为U-I交互，右下角为I-U交互
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print(
                'check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        # TODO 两个adj_mat分别什么用处
        norm_adj_mat = normalized_adj_single(
            adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        # 已抛弃 这里的负样本只要求没有在训练集中出现，没有保证也不再测试集里出现
        # 所以这里所谓的负样本其实可能是测试集中的正样本。
        # 所以没有使用negative-pool，而是重写了sample中的sample_neg_items_for_u
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) -
                             set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        # 先采样user， 再采样 正 负的item
        # for bpr loss
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users)
                     for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                # 防止采样到正样本 & 重复负样本
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items
        # 废弃

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        # 每个user只采样一个 正，负 样本
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test,
                                                        (self.n_train + self.n_test) / (
                                                            self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid)
                                       for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (
                    n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (
                    n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

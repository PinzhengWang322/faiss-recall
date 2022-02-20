from logging.config import valid_ident
import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from preprocess import Dataset
import time

# sampler for batch generation
np.random.seed(100)
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, itemset, batch_size, maxlen, result_queue, SEED, neg_nums):
    def sample():
        
        users = list(user_train.keys())
        user = random.choice(users)
        while len(user_train[user]) <= 1: user = random.choice(users)

        seq = np.zeros([maxlen], dtype=np.int64)
        

        idx = maxlen - 1

        ts = set(user_train[user])

        end = len(user_train[user]) - 1
        if end > maxlen + 1: end = random.randint(maxlen + 1, end)

        seq_len = 0

        for i in reversed(user_train[user][:end]):
            seq_len += 1
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        pos = user_train[user][end]

        neg = random.choice(itemset)
        while neg in ts: neg = random.choice(itemset)

        return (user, seq, pos, neg, seq_len) 

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, itemset, batch_size=64, maxlen=10, n_workers=1, neg_nums = 100):
        itemset = list(itemset)
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      itemset,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      neg_nums,
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(args):
    # return user_train, user_valid, user_test, user_sex, train_item, test_item, Bert_dic
    user_train = {}
    user_valid = {}
    user_test = {}
    dataset = Dataset(args)
    user_train = dataset.read_train()
    user_test = dataset.read_test()
    user_sex = dataset.read_user()
    train_item = dataset.get_train_item()
    test_item = dataset.get_test_item()


    Bert_dic = dataset.get_bert_tabel()

    print("Train:")
    print("The number of users is:", len(user_train))
    print('-' * 30)

    print("Test:")
    print("The number of users is:", len(user_test))
    print('-' * 30)


    return user_train, user_valid, user_test, \
        user_sex, train_item, test_item, Bert_dic,


def produce_test(u, test, args, Bert_dic):
    
    if  len(test[u]) < 2: return
    # seq = np.zeros([args.maxlen], dtype=np.int32)
    idx = args.maxlen - 1
    
    for i in reversed(test[u][:-1]):
        seq = i
        idx -= 1
        if idx == -1: 
            break
    pos = test[u][-1]
    
    # seq = seq[0]
    seq = np.array([Bert_dic[seq][1]],dtype='float32')
    return seq, pos

def produce_test_equal(u, test, args, Bert_dic):
    if  len(test[u]) < 2: return
    
    idx = 10
    seq = np.array([Bert_dic[test[u][-2]][1]],dtype='float32')
    idx -= 1
    seq_len = 1
    for i in reversed(test[u][:-2]):
        seq += np.array([Bert_dic[i][1]],dtype='float32')
        seq_len += 1
        idx -= 1
        if idx == -1: 
            break
    seq /= seq_len
    pos = test[u][-1]
    return seq, pos
    


def recall_test(model, test, args, dim2item):
    HT = 0.0
    all_time = 0
    valid_user = 0.0
    
    users = test.keys()
    
    for u in users:
        if  len(test[u]) < 2: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        seq_len = 0
        for i in reversed(test[u][:-1]):
            seq_len += 1
            seq[idx] = i
            idx -= 1
            if idx == -1: 
                break
        
        rated = set(test[u])
        rated.add(0)
        pos = test[u][-1]
        rated.remove(pos)
        
        t1 = time.time()
        
        recall_items = model.recall([u], seq, [seq_len],recall_num,rated)

        t2 = time.time()
        
        all_time += t2 - t1
        valid_user += 1
        if int(pos) in recall_items: HT += 1
        # break
        
    return print(HT,valid_user, HT / valid_user, all_time)


def evaluate_test2(model, test ,all_items, args, final_dic):
    # self, u, seq, item_ids, final_dic, seq_len
    NDCG_1, NDCG_5, NDCG_200, NDCG_1000 = 0.0, 0.0, 0.0, 0.0
    HT_1, HT_5, HT_200, HT_1000 = 0.0, 0.0, 0.0, 0.0
    all_time = 0
    valid_user = 0.0
    
    users = test.keys()
    # 随机采样部分用户来评测
    for u in users:
        if  len(test[u]) < 2: continue
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        seq_len = 0
        for i in reversed(test[u][:-1]):
            seq_len += 1
            seq[idx] = i
            idx -= 1
            if idx == -1: 
                break
        
        rated = set(test[u])
        rated.add(0)
        item_idx = [test[u][-1]]
        
        for t in all_items:
            if t in rated: continue
            item_idx.append(t)
        
        # item_idx正项 + 100个随机负采样
        t1 = time.time()
        predictions = -model.predict([u], seq, item_idx, final_dic, [seq_len])

        # predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()
        t2 = time.time()
        all_time += t2 - t1
        valid_user += 1

        if rank < 1:
            NDCG_1 += 1 / np.log2(rank + 2)
            HT_1 += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1

        if rank < 200:
            NDCG_200 += 1 / np.log2(rank + 2)
            HT_200 += 1

        if rank < 1000:
            NDCG_1000 += 1 / np.log2(rank + 2)
            HT_1000 += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
        # break

    return (NDCG_1 / valid_user, NDCG_5 / valid_user, NDCG_200 / valid_user, NDCG_1000 / valid_user,\
    HT_1 / valid_user, HT_5 / valid_user, HT_200 / valid_user, HT_1000 / valid_user), all_time



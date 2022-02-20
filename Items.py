import os
import time
import torch
import argparse
import random
import numpy as np

class Items():
    def __init__(self):
        self.dim2id = []
        self.id2dim = {}
        self.x_np = None
        self.Bert_dic = {}

    def upload_train(self, Bert_dic):
        self.Bert_dic = Bert_dic
        x_np = []
        for k in Bert_dic:
            self.id2dim[k] = len(self.dim2id)
            self.dim2id.append(k)
            x_np.append(np.array([Bert_dic[k][1]], dtype='float32'))
        x_np = np.concatenate(x_np)
        self.x_np = x_np
        return
  
    def update(self, id, Bert_emb):
        if id in self.id2dim:
            return 0
        else:
            self.Bert_dic[id] = [0,Bert_emb.tolist()[0],0]
            self.id2dim[id] = len(self.dim2id)
            self.dim2id.append(id)
            # print(self.x_np.shape, Bert_emb.shape)
            # self.x_np = np.concatenate([self.x_np, Bert_emb])
            return 1




import numpy as np
import argparse
import numpy as np
import torch 
from Bert.bert import bert_encoder
from transformers import AutoModel, AutoTokenizer
from preprocess import Dataset
import os



def read_content(content_dic_path, args):
    content_dic = {}
    f = open(content_dic_path, 'r')
    lines = f.readlines()
    for line in lines[1:]:
        line = line.split('\t')
        cid = int(line[0])
        content_dic[cid] = []
        if args.use_ctitle:
            content_dic[cid].append(line[1][:300])
        if args.use_desc:
            content_dic[cid].append(line[2][:300])
        if args.use_ttitle:
            content_dic[cid].append(line[4][:300])
    return content_dic

class content2bert:
    def __init__(self,args):
        self.dev = args.device
        model_dir = args.Bert_folder_path
        model = AutoModel.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.model = model.to(self.dev)
    
    def make_bert(self,content):
        return np.array([bert_encoder([content], self.model, self.tokenizer, self.dev)], dtype='float32')
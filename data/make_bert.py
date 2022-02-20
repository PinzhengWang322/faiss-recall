import numpy as np
import argparse
import sys 
sys.path.append("..") 
import numpy as np
import torch 
from Bert.bert import bert_encoder
from transformers import AutoModel, AutoTokenizer
from preprocess import Dataset
import os

def make_bert(dic, item_dic):
    model_dir = "../Bert/chinese_roberta_wwm_ext_pytorch"
    model = AutoModel.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = model.to('cuda')
    cnt = 0
    new_dic = {}
    for k in item_dic:
        if cnt % 100 == 1:
            print(cnt)
        cnt += 1
        new_dic[k] = []
        for id in [0,1,2]:
            new_dic[k].append(bert_encoder([dic[k][0][id]], model, tokenizer, 'cuda'))
            # new_dic[k].append([dic[k][0][id]])
        new_dic[k].append(dic[k][1])
    np.save('save.npy', new_dic) 


def read_content(path, args):
        f = open(path, 'r')
        lines = f.readlines()
        dic = {}
        for line in lines[1:]:
            line = line.split('\t')
            cid = int(line[0])
            dic[cid] = [[],[]]
            if args.use_ctitle:
                dic[cid][0].append(line[1][:300])
            if args.use_desc:
                dic[cid][0].append(line[2][:300])
            if args.use_ttitle:
                dic[cid][0].append(line[4][:300])
            if args.use_praise:
                dic[cid][1].append(int(line[-7]))
            if args.use_reply:
                dic[cid][1].append(int(line[-6]))
            if args.use_foward:
                dic[cid][1].append(int(line[-5]))
        return dic
        
def pca(dic_path, num):
    dic = np.load(dic_path,allow_pickle=True).item()
    keys = dic.keys()
    lst = []
    
    for k in keys:
        lst[0]=dic[k][0]
        lst[1]=dic[k][0]

    X = np.array(lst)
    print(X.shape)
    pca = PCA(n_components=num)
    pca.fit(X)
    X_new = pca.transform(X)
    print(X_new.shape)
    new_dic = {}
    for id,k in enumerate(keys):
        new_dic[k] = X_new[id].tolist()
    np.save('save_bert' + str(num) + '.npy', new_dic) 
    return
if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.use_ctitle = True
    # args.use_desc = True
    # args.use_ttitle = True
    # args.use_praise = True
    # args.use_reply = True
    # args.use_foward = True
    # dic = read_content('content.txt', args)
    # args.train_data_path = 'train.txt'
    # args.test_data_path = 'test.txt'
    # args.user_path = 'user.txt'
    # args.use_bert = False
    # dataset = Dataset(args)
    # dataset.read_train()
    # item_dic = dataset.get_train_item()
    # make_bert(dic, item_dic)
    new_dic = np.load('save.npy',allow_pickle=True).item()
    uid = 776212
    print(new_dic[uid][0])
    print("Done!")

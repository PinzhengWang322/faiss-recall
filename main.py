import os
import time
import torch
import argparse
import random
import numpy as np

from utils import *
import time
from item_emb import content2bert, read_content
from Items import Items
import faiss                   


random.seed(10)
torch.manual_seed(20)
torch.cuda.manual_seed_all(30)
np.random.seed(10)
torch.backends.cudnn.deterministic = True


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

random.seed( 10 )

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', default="./Log")
parser.add_argument('--name', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=1, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_epochs', default=5, type=int)
parser.add_argument('--dropout_rate', default=0, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--neg_nums', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--use_bert',default=True,type=bool)
parser.add_argument('--sex_dim',default=50,type=bool)
parser.add_argument('--bert_dim',default=768,type=bool)



args = parser.parse_args()
args.bert_path = 'data/new_save.npy'
args.train_data_path = 'data/train.txt'
args.test_data_path = 'data/test.txt'
args.user_path = 'data/user.txt'
args.use_ctitle = False
args.use_desc = True
args.use_ttitle = False
args.use_praise = True
args.use_reply = True
args.use_foward = True
args.Bert_folder_path = "Bert/chinese_roberta_wwm_ext_pytorch"
args.content_dic_path = "data/content.txt"
args.use_c = 1

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.name = "./" + args.name

if not os.path.isdir(args.log_path + '/' + args.name):
    os.makedirs(args.log_path + '/' + args.name)
with open(os.path.join(args.log_path + '/' + args.name, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    BEGIN = time.time()
    dataset = data_partition(args)
    user_train, user_valid, user_test, user_sex, train_item,  test_item, Bert_dic = dataset
    num_batch = len(user_train) // args.batch_size  
    Bert_dic[0] = [[0 for i in range(args.bert_dim)] for j in range(3)]
    items = Items() 
    items.upload_train(Bert_dic) # 上传之前预处理好的Bert_dic
    xtrain = items.x_np # 用于faiss训练的矩阵，暴力搜索不需要
    print(xtrain.shape)

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(args.bert_dim) #运用L2距离
    # index = faiss.IndexFlatIP(args.bert_dim)
    index = faiss.index_cpu_to_gpu(res, 0, index) #上传到GPU上
    index.add(xtrain) 
    print(index.ntotal)

    content_dic = read_content(args.content_dic_path, args) #读入文本内容
    c2b = content2bert(args) 

    all_items = list(set.union(train_item, test_item))
    print(len(all_items))
    
    t1 = time.time()
    for id in all_items:
        if id not in items.id2dim:
            
            bert_emb = c2b.make_bert(content_dic[id][0]) #训练集中未出现的item进行Bert编码
            index.add(bert_emb) 
            items.update(id, bert_emb) #更新
            
   
    t2 = time.time()
    print('all_time',t2 - t1)
    print(index.ntotal)
    
    print('evaluate:','_' * 30)
    HT = 0.0
    user_num = 0.0
    all_time = 0.0
    
    for u in  user_test: #评测效果
        user_num += 1
        t1 = time.time()
        testx, testy = produce_test(u, user_test, args, items.Bert_dic) #用倒数第二的项目预测倒一位置概率
        _ , I = index.search(testx, 200)
        I = I[0]
        Recall_items = [items.dim2id[i] for i in I]
        t2 = time.time()
        all_time += t2 - t1
        
        if testy in Recall_items:
            HT += 1
    print(HT, user_num, HT / user_num, all_time)

    f.close()
    print("Done")

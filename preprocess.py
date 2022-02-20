import argparse
import time
import numpy as np

class Dataset():
    def __init__(self, args):
        self.args = args
        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        # self.content_path = args.content_path
        self.user_path= args.user_path
        self.user_test = {}
        self.train_item = set()
        self.test_item = set()
        
        if args.use_bert:
            t1 = time.time()
            self.item_Bert = np.load(args.bert_path,allow_pickle=True).item()
            t2 = time.time()
            print("Loading Bert time:", t2 - t1, 's')

    def read_user(self):
        user_sex = {}
        f = open(self.user_path, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split('\t')
            if line[0][-1] not in '1234567890': continue
            user_sex[int(line[0])] = int(line[1])
        f.close()
        return user_sex

    def read_train(self):
        user_train = {}
        f = open(self.train_data_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) < 3: continue
            uid = int(line[0])
            user_train[uid] = []
            befi = '0'
            for item in line[1:]:
                if item == '0' or item == befi: continue 
                user_train[uid].append(int(item))
                self.train_item.add(int(item))
                befi = item
        f.close()
        return user_train

    def read_test(self):
        user_test = {}
        f = open(self.test_data_path, 'r')
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if len(line) < 3: continue
            uid = int(line[0])
            user_test[uid] = []
            for item in line[1:]:
                if item == '0': continue 
                user_test[uid].append(int(item))
                self.test_item.add(int(item))
        f.close()
        return user_test

    def get_bert_tabel(self):
        return self.item_Bert

    def get_train_item(self):
        return self.train_item

    def get_test_item(self):
        return self.test_item
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test_for_process')
    args = parser.parse_args()
    args.train_data_path = 'data/train.txt'
    args.test_data_path = 'data/test.txt'
    args.user_path = 'data/user.txt'
    args.use_bert = False
    dataset = Dataset(args)
    user_sex = dataset.read_user()
    user_train = dataset.read_train()
    user_test = dataset.read_test()
    s = set()
    for i in user_sex:
        s.add(user_sex[i])
    print(s)
    pass
import sys
import torch
from collections import Counter
import torch.utils.data as data
import random
import pickle
import numpy as np
import h5py
from tqdm import tqdm
import json
from scipy import sparse
from transformers import AutoTokenizer
import javalang
from os import listdir
from os.path import isfile, join
import json
import subprocess
import re
import os
import copy
import math
from transformers import RobertaTokenizer
model_type = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_type)
class dotdict(dict):
    def __getattr__(self, name):  
        return self[name]

space_token = 'Ġ'
args = dotdict({
    'lr':1e-4,
    'model_type':model_type,
    'max_conflict_length':500,
    'max_resolve_length':200,
})
class Dataset(data.Dataset):
    def __init__(self, args, tokenizer, data_name):
        self.max_conflict_length = args.max_conflict_length
        self.max_resolve_length = args.max_resolve_length
        self.tokenizer = tokenizer


        data_path = 'processed_%s.pkl'%(data_name) 

        if not os.path.exists(data_path):
            total_raw_data_path = 'RAW_DATA'
            all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(open('%s/raw_data'%(total_raw_data_path)))

            self.process_data(all_raw_base, all_raw_a, all_raw_b, all_raw_res, data_name)
        self.data = pickle.load(open(data_path, "rb"))


    def process_data(self, all_raw_base, all_raw_a, all_raw_b, all_raw_res,  data_name="train"):
        data_num = len(all_raw_base)
        each_num = 1000
        
        total_batches = []

        for i in tqdm(range(math.ceil(data_num / each_num))):
            cur_start = i * each_num
            cur_end = min((i + 1) * each_num, data_num)
            cur_data_path = 'PROCESSED/processed_%s_%s.pkl'%(cur_start, cur_end)
            self.ii = i
            cur_batches = pickle.load(open(cur_data_path, 'rb'))
            if len(total_batches) == 0:
                total_batches = cur_batches
            else:
                for i in range(len(total_batches)):
                    total_batches[i] = np.concatenate([total_batches[i], cur_batches[i]], axis = 0)


        print('all data num:%d remaining num:%d'%(data_num, len(total_batches[0])))
        assert data_num == len(total_batches[0])
        num_train = int(data_num * 0.8)
        num_valid = int(data_num * 0.1)
        num_test = data_num - num_train - num_valid

        index = list(range(data_num))
        random.shuffle(index)
        train_index = index[:num_train]
        valid_index = index[num_train:num_train + num_valid]
        test_index = index[num_train + num_valid:]

        all_index = {'train':train_index, 'valid': valid_index, 'test': test_index}
        json.dump(all_index, open('all_index','w'))
        
        train_batches = []
        valid_batches = []
        test_batches = []

        for i in range(len(total_batches)):
            train_batches.append(total_batches[i][train_index])
            valid_batches.append(total_batches[i][valid_index])
            test_batches.append(total_batches[i][test_index])  
        pickle.dump(train_batches, open("processed_train.pkl", 'wb'))
        pickle.dump(valid_batches, open("processed_valid.pkl", 'wb'))
        pickle.dump(test_batches, open("processed_test.pkl", 'wb'))


    def pad_length(self, tokens, max_length, pad_id):
        if len(tokens) <= max_length:
            tokens = tokens + [pad_id] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        assert len(tokens) == max_length
        return tokens

    
    def __getitem__(self, offset):
        data = []
        for i in range(len(self.data)):
            if type(self.data[i][offset]) == np.ndarray:
                data.append(self.data[i][offset])
            else:
                data.append(self.data[i][offset].toarray())  
        return data
    def __len__(self):
        return len(self.data[0])





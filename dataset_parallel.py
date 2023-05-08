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
from transformers import RobertaTokenizer, T5Model, T5ForConditionalGeneration, AdamW
model_type = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_type)
class dotdict(dict):
    def __getattr__(self, name):  
        return self[name]

space_token = 'Ġ'
args = dotdict({
    'model_type':model_type,
    'max_conflict_length':500,
    'max_resolve_length':200,
})


brackets_tokens = ['<lbra>', '<mbra>', '<rbra>']
succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)

class Dataset(data.Dataset):
    def __init__(self, args, tokenizer, process_start, process_end):
        self.max_conflict_length = args.max_conflict_length
        self.max_resolve_length = args.max_resolve_length
        self.tokenizer = tokenizer
        self.start = process_start
        self.end = process_end
        self.lbra_token = '<lbra>'
        self.rbra_token = '<rbra>'


        if not os.path.exists('PROCESSED'):
            os.mkdir('PROCESSED')
        self.data_path = 'PROCESSED/processed_%s_%s.pkl'%(self.start, self.end)

        total_raw_data_path = 'RAW_DATA'
        all_raw_base, all_raw_a, all_raw_b, all_raw_res = json.load(open('%s/raw_data'%(total_raw_data_path)))
        

        self.process_data(all_raw_base, all_raw_a, all_raw_b, all_raw_res)


    
    def process_data(self, all_raw_base, all_raw_a, all_raw_b, all_raw_res):
        data_num = self.end - self.start
        max_conflict_length = 0
        max_resolve_length = 0
        inputs = []
        outputs = []

        for i in tqdm(range(self.start, self.end)):

            self.ii = i
            raw_base = all_raw_base[i]
            raw_a = all_raw_a[i]
            raw_b = all_raw_b[i]
            raw_res = all_raw_res[i]

            
            raw_base = ' '.join(raw_base.split())
            raw_a = ' '.join(raw_a.split())
            raw_b = ' '.join(raw_b.split())
            raw_res = ' '.join(raw_res.split())

            tokens_base = self.tokenizer.tokenize(raw_base)
            tokens_a = self.tokenizer.tokenize(raw_a)
            tokens_b = self.tokenizer.tokenize(raw_b)
            tokens_res = self.tokenizer.tokenize(raw_res)


            tokens_input = self.git_merge(tokens_base, tokens_a, tokens_b)

            ids_input = self.tokenizer.convert_tokens_to_ids(tokens_input)
            ids_res = self.tokenizer.convert_tokens_to_ids(tokens_res)

            cur_input = ids_input
            cur_output = [self.tokenizer.bos_token_id] + ids_res + [self.tokenizer.eos_token_id]

                

            max_conflict_length = max(max_conflict_length, len(cur_input))
            max_resolve_length = max(max_resolve_length, len(cur_output))


            cur_input = self.pad_length(cur_input, self.max_conflict_length, self.tokenizer.pad_token_id)
            cur_output = self.pad_length(cur_output, self.max_resolve_length, self.tokenizer.pad_token_id)
            inputs.append(cur_input)
            outputs.append(cur_output)
  
        print('max_conflict_length, max_resolve_length', max_conflict_length, max_resolve_length)
        print('all data num:%d remaining num:%d'%(data_num, len(inputs)))
        assert data_num == len(inputs)
        data_num = len(inputs)
        batches = [np.array(inputs), np.array(outputs)]

        pickle.dump(batches, open(self.data_path, 'wb'))


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

    
    def git_merge(self, tokens_base, tokens_a, tokens_b):
        merge_path = 'GIT_MERGE_FILES/%s'%self.ii
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)
        with open('%s/base'%merge_path, 'w') as f:
            f.write('\n'.join(tokens_base))
        with open('%s/a'%merge_path, 'w') as f:
            f.write('\n'.join(tokens_a))
        with open('%s/b'%merge_path, 'w') as f:
            f.write('\n'.join(tokens_b))
        
        final_tokens = []
        self.execute_command('git merge-file -L a -L base -L b %s/a %s/base %s/b --diff3 -p > %s/merge'%(merge_path, merge_path, merge_path, merge_path))
        merge_res = open('%s/merge'%merge_path).read().splitlines()
        merge_res = [x.strip() for x in merge_res if x.strip()]
        format_ids = [k for k, x in enumerate(merge_res) if x == '<<<<<<< a' or x == '>>>>>>> b' or x == '||||||| base' or x == '=======']
        assert len(format_ids) % 4 == 0
        start = 0
        for k, x in enumerate(format_ids):
            if k % 4 == 0:
                assert merge_res[format_ids[k]] == '<<<<<<< a' and merge_res[format_ids[k + 1]] =='||||||| base' and merge_res[format_ids[k + 2]] == '=======' and merge_res[format_ids[k + 3]] == '>>>>>>> b'
                context_tokens = merge_res[start:format_ids[k]]
                a_tokens = merge_res[format_ids[k] + 1:format_ids[k + 1]]
                base_tokens = merge_res[format_ids[k + 1] + 1:format_ids[k + 2]]
                b_tokens = merge_res[format_ids[k + 2] + 1:format_ids[k + 3]]
                start = format_ids[k + 3] + 1

                final_tokens += context_tokens + [self.lbra_token] + a_tokens + [self.tokenizer.sep_token] + base_tokens + [self.tokenizer.sep_token] + b_tokens + [self.rbra_token]
                



        if start != len(merge_res):
            final_tokens += merge_res[start:]
        final_tokens = [self.tokenizer.bos_token] + final_tokens + [self.tokenizer.eos_token]
        
        return final_tokens
    def execute_command(self, cmd):
        p = subprocess.Popen(cmd, shell=True)
        p.wait()


if __name__ == '__main__':
    process_start = int(sys.argv[1])
    process_end = int(sys.argv[2])
    dataset = Dataset(args, tokenizer, process_start, process_end)




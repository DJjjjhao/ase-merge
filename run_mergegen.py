from cmath import inf
import os
from transformers import RobertaTokenizer, T5Model, T5ForConditionalGeneration, AdamW
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from torch import optim
from torch.nn import functional as F
import torch
import torch.nn as nn
import json
import numpy as np
import random
from dataset import Dataset
import sys 
from torch.utils.data import DataLoader
import math
from accelerate import Accelerator
import time
from tqdm import tqdm
from accelerate import DistributedDataParallelKwargs
space_token = 'Ġ'
stage = str(sys.argv[1])


model_type = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_type)

brackets_tokens = ['<lbra>', '<mbra>', '<rbra>']
succeed_num = tokenizer.add_tokens(brackets_tokens)
assert succeed_num == len(brackets_tokens)

accelerator = Accelerator()
beam_num = 3

use_cuda = torch.cuda.is_available()
device_ids = list(range(torch.cuda.device_count()))


class dotdict(dict):
    def __getattr__(self, name):  
        return self[name]

args = dotdict({
    'batch_size':35,
    'test_batch_size':30,
    'epoches':100,
    'lr':1e-4,
    'model_type':model_type,
    'max_conflict_length':500,
    'max_resolve_length':200,
})

def seed_everything(seed=0):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_model(model,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path)

def get_tensor(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)

    return tensor

class MergeT5(nn.Module):
    def __init__(self, args):
        super(MergeT5, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(args.model_type)
        self.t5.resize_token_embeddings(len(tokenizer))
        self.embedding_dim = self.t5.config.hidden_size

    def forward(self, input_txt, output_txt, stage):
        attention_mask = input_txt != tokenizer.pad_token_id
        
        input_em = self.t5.encoder(input_ids=input_txt, attention_mask=attention_mask, return_dict=True)['last_hidden_state']
        logits = self.t5.decoder(input_ids=output_txt, encoder_hidden_states=input_em, encoder_attention_mask=attention_mask, return_dict=True)['last_hidden_state']
        logits = logits * (self.embedding_dim ** -0.5)
        logits = self.t5.lm_head(logits)
        
        outputs = F.softmax(logits, dim=-1)
        outputs = torch.log(outputs.clamp(min=1e-10, max=1))

        label = output_txt
        label = torch.cat([label, torch.ones(len(label), 1).cuda(input_txt.device) * tokenizer.pad_token_id], dim=-1)
        label = label[:,1:]
        label = label.long()
        mask = label != 0
        
        loss = F.nll_loss(outputs.view(-1, outputs.size(-1)), label.contiguous().view(-1), 
        reduction = 'none')
        loss = loss.masked_fill(mask.view(-1)==False, 0)
        if stage == 'train':
            return loss.sum(), mask.sum()
        elif stage == 'dev' or stage == 'test':
            return torch.argmax(outputs, dim=-1), loss.sum(), mask.sum(), label
def train(accelerator, model, train_loader, optimizer, epoch, best_acc, dev_loader, f):
    model.train()
    total_data = 0
    total_loss = 0

    for idx, batch in enumerate(tqdm(train_loader)):
        
        assert isinstance(batch, list)
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])

        
        loss, mask = model(batch[0], batch[1], 'train')
        loss = loss.sum() / mask.sum()

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        total_data += len(batch[0])
        total_loss += loss.item()

    accelerator.print("epoch: %d batch: %d/%d  data: %d/%d loss: %.4f device:%s\n"%(epoch, idx, len(train_loader), total_data, len(train_loader.dataset), total_loss / len(train_loader), loss.device))
    flag_bigger = None
    if accelerator.is_main_process:

        exactly_match_num, exactly_match_ids, total_dev_output = dev(model, dev_loader, epoch)
        exactly_match_ratio = exactly_match_num / len(dev_loader.dataset)
        f.write('epoch: {} exactly match:{} is better: {}\n'.format(epoch, exactly_match_num / len(dev_loader.dataset), exactly_match_ratio > best_acc))
        f.flush()
        if not os.path.exists('OUTPUT/DEV_OUTPUT'):
            os.makedirs('OUTPUT/DEV_OUTPUT')
        flag_bigger = exactly_match_ratio > best_acc
        if exactly_match_ratio > best_acc:
            best_acc = exactly_match_ratio
            torch.save(model.module.state_dict(),"best_model.pt")
            for k in range(len(total_dev_output)):
                p = open('OUTPUT/DEV_OUTPUT/%s'%(k),'w')
                p.write(total_dev_output[k])
                p.close()

    accelerator.wait_for_everyone()



    return best_acc, flag_bigger

def see_results(inputs, outputs, targets):
    assert len(outputs.shape) == 3

    inputs = inputs.cpu().numpy()
    if len(outputs.shape) == 2:
        outputs = outputs.unsqueeze(1)
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    output_strings = []
    target_strings = []
    for i in range(len(outputs)):
        beam_output_strings = []
        for j in range(len(outputs[i])):
            cur_output = outputs[i][j].tolist()
            if tokenizer.eos_token_id in cur_output:
                cur_output = cur_output[:cur_output.index(tokenizer.eos_token_id)] 
            output_token = tokenizer.convert_ids_to_tokens(cur_output)
            output_string = ''.join(output_token).replace(space_token, " ")
            beam_output_strings.append(output_string)

        cur_target = targets[i].tolist()
        cur_target = cur_target[1:]
        if tokenizer.eos_token_id in cur_target:
            cur_target = cur_target[:cur_target.index(tokenizer.eos_token_id)] 
        
        ref_token = tokenizer.convert_ids_to_tokens(cur_target)
        ref_string = ''.join(ref_token).replace(space_token, " ")

    
        output_strings.append(beam_output_strings)
        target_strings.append(ref_string)
    return output_strings, target_strings


def dev(model, val_loader, epoch, dev_type='train'):
    all_index = json.load(open('all_index'))
    valid_index = all_index['valid']
    gen_cannot_tokenize_num = 0
    ref_cannot_tokenize_num = 0
    model.eval()
    total_dev_output = []
    
    total_data = 0
    total_loss = 0
    total_mask = 0
    exactly_match_num = 0
    exactly_match_ids = []
    total_results = None
    total_label = None

    total_input_strings = []
    total_output_strings = []
    total_target_strings = []
    total_output_tokens = []
    total_target_tokens = []
    for idx, batch in enumerate(tqdm(val_loader)):
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
            batch[i] = batch[i].cuda()
        
        with torch.no_grad():   
            output, loss, mask, label = model(batch[0], batch[1], 'dev')
            results = output == label
            if idx == 0:
                total_results = results.cpu()
                total_label = label.cpu()
            else:
                total_results = torch.cat((total_results, results.cpu()), dim=0)
                total_label = torch.cat((total_label, label.cpu()), dim=0)
            total_loss += loss.sum().item()
            total_mask += mask.sum().item()
            
            if dev_type == 'dev':
                # -----------------see succeed/fail data-----------------
                output_strings, ref_strings,  = see_results(batch[0], output, batch[1])
                total_output_strings.extend(output_strings)
                total_target_strings.extend(ref_strings)
                # -----------------see succeed/fail data-----------------
        total_data += len(output)

    total_results = torch.masked_fill(total_results, total_label == 0, True)
    total_results = torch.all(total_results, dim=-1)
    exactly_match_ids = total_results.nonzero(as_tuple=False).tolist()
    exactly_match_num = total_results.sum().item()

    assert total_data == len(val_loader.dataset)


    if dev_type == 'dev':
        # -----------------see succeed/fail data-----------------
        output_path = 'OUTPUT/DEV_OUTPUT'
        if not os.path.exists('%s/SUCCEED'%output_path):
            os.makedirs('%s/SUCCEED'%output_path)
            os.makedirs('%s/FAIL'%output_path)
        for i in range(len(total_results)):
            if total_results[i]:
                p = open('%s/SUCCEED/%s'%(output_path, valid_index[i]), 'w')
            else:
                p = open('%s/FAIL/%s'%(output_path, valid_index[i]), 'w')

            
            p.write('---------------------------------\n')
            p.write('ref_dataloader:\n%s\n'%total_target_strings[i])
            p.write('---------------------------------\n')
            p.write('output:\n%s\n'%total_output_strings[i])
            p.write('---------------------------------\n')
            p.close()
            p.close()
        # -----------------see succeed/fail data-----------------        
        total_dev_output = total_output_strings
    return exactly_match_num, exactly_match_ids, total_dev_output



def test_beam(model, test_loader):
    all_index = json.load(open('all_index'))
    test_index = all_index['test']
    gen_cannot_tokenize_num = 0
    ref_cannot_tokenize_num = 0
    model.eval()
    total_test_output = []
    
    total_data = 0
    total_loss = 0
    total_mask = 0
    exactly_match_num = 0
    exactly_match_ids = []
    total_results = None
    total_label = None

    total_input_strings = []
    total_output_strings = []
    total_target_strings = []
    total_output_tokens = []
    total_target_tokens = []
    for idx, batch in enumerate(tqdm(test_loader)):
        for i in range(len(batch)):
            batch[i] = get_tensor(batch[i])
            batch[i] = batch[i].cuda()
        with torch.no_grad():   
            input_attention_mask = batch[0] != tokenizer.pad_token_id
            input_em = model.t5.encoder(input_ids=batch[0], attention_mask=input_attention_mask, return_dict=True)['last_hidden_state']
            input_em = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=input_em)
            beam_output = model.t5.generate(encoder_outputs=input_em, attention_mask=input_attention_mask, decoder_input_ids=(torch.ones(len(batch[0]), 1) * tokenizer.bos_token_id).long().cuda(), num_beams=beam_num, num_return_sequences=beam_num, max_new_tokens=args.max_resolve_length * 2)
            beam_output = beam_output[:,1:]

            beam_output = beam_output.view(len(batch[0]), beam_num, -1)
            output = beam_output[:,0,:]

            output = output[:,:args.max_resolve_length]
            output = torch.cat([output, torch.zeros(len(output), args.max_resolve_length - len(output[0])).long().cuda()], dim=-1)

            label = batch[1]
            label = torch.cat([label, torch.zeros(len(label), 1).long().cuda(batch[0].device)], dim=-1)
            label = label[:,1:]

            assert output.shape == label.shape


            results = output == label
            if idx == 0:
                total_results = results.cpu()
                total_label = label.cpu()
            else:
                total_results = torch.cat((total_results, results.cpu()), dim=0)
                total_label = torch.cat((total_label, label.cpu()), dim=0)

            
            # -----------------see succeed/fail data-----------------
            output_strings, ref_strings = see_results(batch[0], beam_output, batch[1])
            total_output_strings.extend(output_strings)
            total_target_strings.extend(ref_strings)
            # -----------------see succeed/fail data-----------------
        total_data += len(output)

    total_results = torch.masked_fill(total_results, total_label == 0, True)
    total_results = torch.all(total_results, dim=-1)
    exactly_match_ids = total_results.nonzero(as_tuple=False).tolist()
    exactly_match_num = total_results.sum().item()

    assert total_data == len(test_loader.dataset)


    # -----------------see succeed/fail data-----------------
    output_path = 'OUTPUT/test_OUTPUT'
    if not os.path.exists('%s/SUCCEED'%output_path):
        os.makedirs('%s/SUCCEED'%output_path)
        os.makedirs('%s/FAIL'%output_path)
    for i in range(len(total_results)):
        if total_results[i]:
            p = open('%s/SUCCEED/%s'%(output_path, test_index[i]), 'w')
        else:
            p = open('%s/FAIL/%s'%(output_path, test_index[i]), 'w')

        p.write('ref_dataloader:\n%s\n'%total_target_strings[i])
        p.write('---------------------------------\n')
        p.write('output:\n%s\n'%total_output_strings[i])
        p.write('---------------------------------\n')
        p.close()
        p.close()
    # -----------------see succeed/fail data-----------------        
    total_test_output = total_output_strings
    return exactly_match_num, exactly_match_ids, total_test_output


def main_train():
    
    open('train_state', 'w').write(str(1))
    if accelerator.is_main_process:
        train_set = Dataset(args, tokenizer, 'train')
        dev_set = Dataset(args, tokenizer, 'valid')
    accelerator.wait_for_everyone()
    
    train_set = Dataset(args, tokenizer, 'train')
    dev_set = Dataset(args, tokenizer, 'valid')
    start_time = time.time()
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.test_batch_size)
    model = MergeT5(args)
    optimizer = AdamW(model.parameters(), args.lr)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if os.path.exists("best_model.pt"):
        model.module.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
        print('model loaded!')
    if accelerator.is_main_process:
        if os.path.exists("best_model.pt"):
            exactly_match_num, _, _ = dev(model, dev_loader, -1)
            best_acc = exactly_match_num / len(dev_loader.dataset)
        else:
            best_acc = -1
        open('best_acc', 'w').write(str(best_acc))
    accelerator.wait_for_everyone()
    best_acc = float(open('best_acc', 'r').read())
    print(best_acc)

    f = open('OUTPUT/train_process','a')
    
    small_num = 0
    try_num = 0
    max_try_num = 3
    max_small_num = 6
    for epoch in range(args.epoches):
        if int(open('train_state').read()) == 0:
            break
        best_acc, flag_bigger = train(accelerator, model, train_loader, optimizer, epoch, best_acc, dev_loader, f)
        if accelerator.is_main_process:
            if flag_bigger == True:
                small_num = 0
            else:
                small_num += 1
                if small_num == max_small_num:
                    small_num = 0
                    try_num += 1
                    model.module.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.5
                if try_num == max_try_num:
                    open('train_state', 'w').write(str(0))
        accelerator.wait_for_everyone()
    accelerator.wait_for_everyone()
    end_time = time.time()
    if accelerator.is_main_process:
        f.write("time: %sh"%((end_time - start_time) / 3600))
        f.close()

def main_dev():
    dev_set = Dataset(args, tokenizer, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)
    model = MergeT5(args)
    model.load_state_dict(torch.load("best_model.pt"))
    if use_cuda:
        model = nn.DataParallel(model, device_ids = device_ids)
        model = model.cuda(device_ids[0])
    best_acc = -1
    exactly_match_num, exactly_match_ids, total_dev_output = dev(model, dev_loader, -1, dev_type='dev')

    json.dump(total_dev_output, open('OUTPUT/total_gen_output_dev.json', 'w'))

    with open('OUTPUT/dev_process', 'w') as f:
        f.write('exactly match: %f\n'%(exactly_match_num / len(dev_loader.dataset)))
    
        
def main_test():
    dev_set = Dataset(args, tokenizer, 'valid')
    dev_loader = DataLoader(dataset=dev_set, batch_size=args.batch_size)

    test_set = Dataset(args, tokenizer, 'test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size)
    model = MergeT5(args)
    model.load_state_dict(torch.load("best_model.pt"))
    if use_cuda:
        # model = nn.DataParallel(model, device_ids = device_ids)
        model = model.cuda(device_ids[0])
    f =  open('OUTPUT/test_process', 'w')
    dev_exactly_match_num, _, _ = dev(model, dev_loader, -1)
    f.write('dev_exactly_match: %f\n'%(dev_exactly_match_num / len(dev_loader.dataset)))
    f.flush()
    exactly_match_num, exactly_match_ids, total_test_output = test_beam(model, test_loader)

    json.dump(total_test_output, open('OUTPUT/total_gen_output_test_beam.json', 'w'))
    json.dump(exactly_match_ids, open('RESULTS/test_gen_exactly_match_ids', 'w'))

    
        
    f.write('exactly match: %f\n'%(exactly_match_num / len(test_loader.dataset)))
    f.close()

if __name__ == '__main__':
    
    seed_everything(0)
    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    if stage == 'train':
        main_train()
    elif stage == 'dev':
        main_dev()
    elif stage== 'test':
        main_test()


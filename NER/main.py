# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 04:47:34 2020

@author: yyimi
"""

from NER_data import build_dataset
from Batch import BatchManager
from NER_functions import result_to_json,test_ner
from NER_data import prepare_dataset,load_word2id,load_tag2id

import torch
import torch.nn as nn
import torch.optim as optim
from model.model import NERLSTM_CRF

from NER_parameters import params
from logs.logger import logger


import time,os
from datetime import timedelta
from pprint import pprint

config = params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Record training time """
def training_timer(start_time):
    end_time = time.time()
    time_used = end_time - start_time 
    return timedelta(seconds=int(round(time_used))) 

def train():
    
    #load the dataset
    
    train_data,dev_data,test_data,char_to_id, id_to_char, tag_to_id, id_to_tag, emb_matrix = build_dataset()
    logger.info("%i / %i / %i sentences in train / dev / test." 
                % (len(train_data), len(dev_data), len(test_data)))
    
    #Batch
    train_manager = BatchManager(train_data, config.batch_size)
    dev_manager = BatchManager(dev_data, config.batch_size)
    test_manager = BatchManager(test_data, config.batch_size) 
    
    model = NERLSTM_CRF(config, char_to_id, tag_to_id, emb_matrix)
    model.train()
    model.to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
    #lr:learning rate  #weight decay:L2 penalty
    
    #early stop 
    total_batch = 0  
    dev_best_f1 = float('-inf')
    last_improve = 0  
    flag = False     
    
    start_time = time.time()
    logger.info(" Start Training ...... ")
    for epoch in range(config.max_epoch):
        
        logger.info('Epoch [{}/{}]'.format(epoch + 1, config.max_epoch))
        
        for index, batch in enumerate(train_manager.iter_batch(shuffle=True)):
            
            optimizer.zero_grad() #initialize the gradient
            """"Loss and backward propagation"""
            _, char_ids, len_ids, tag_ids, mask = batch
            loss = model.log_likelihood(char_ids,len_ids,tag_ids, mask)
            loss.backward()
            
            
            """ Gradient Clip Maximum: 5 """
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip)
            optimizer.step()
            
            #Check model after certain number of batchs
            if total_batch % config.steps_check == 0:
                
                model.eval()
                dev_f1,dev_loss = evaluate(model, dev_manager, id_to_tag)
                
                """ check f1 value for early stop """
                if dev_f1 > dev_best_f1:
                    
                    evaluate(model, test_manager, id_to_tag, test=True)
                    dev_best_f1 = dev_f1
                    torch.save(model, os.path.join(config.save_dir,"medical_ner.ckpt"))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                    
                time_used = training_timer(start_time)
                msg = 'Iter: {} | Dev Loss: {:.4f} | Dev F1-macro: {:.4f} | Time: {} | {}'
                logger.info(msg.format(total_batch, dev_loss, dev_f1, time_used, improve))  
                
                model.train()
                
            total_batch += 1
            if total_batch - last_improve > config.require_improve:
                """if the f1 on dev dataset does not imporve for more than 5000batches, stop the train""" 
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break                
                

def evaluate_helper(model, data_manager, id_to_tag):
      
    with torch.no_grad():
        
        total_loss = 0
        results = []
        for batch in data_manager.iter_batch():
            
            chars, char_ids, len_ids, tag_ids, mask = batch
            
            batch_paths = model(char_ids, len_ids, mask)
            loss = model.log_likelihood(char_ids, len_ids, tag_ids,mask)
            total_loss += loss.item()    
            
            """ 忽略<pad>标签，计算每个样本的真实长度 """
            lengths = [len([j for j in i if j > 0]) for i in tag_ids.tolist()]
            
            tag_ids = tag_ids.tolist()
            for i in range(len(chars)):
                result = []
                string = chars[i][:lengths[i]]
                
                """ 把id转换为标签 """
                gold = [id_to_tag[int(x)] for x in tag_ids[i][:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]               
                
                """ 用CoNLL-2000的实体识别评估脚本, 需要按其要求的格式保存结果，
                即 字-真实标签-预测标签 用空格拼接"""
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        
        aver_loss = total_loss / (data_manager.len_data * config.batch_size)        
        return results, aver_loss  
    

def evaluate(model, data, id_to_tag, test=False):
    
    """ 得到预测的标签（非id）和损失 """
    ner_results, aver_loss = evaluate_helper(model, data, id_to_tag)
    
    """ 用CoNLL-2000的实体识别评估脚本来计算F1值 """
    eval_lines = test_ner(ner_results, config.save_dir)
    
    if test:
        
        """ 如果是测试，则打印评估结果 """
        for line in eval_lines:
            logger.info(line)
            
    f1 = float(eval_lines[1].strip().split()[-1]) / 100
    
    return f1, aver_loss


def predict(input_str):
    
    char_to_id, id_to_char = load_word2id(config.word2id)
    tag_to_id, id_to_tag = load_tag2id(config.tag2id)
    
    """ 用cpu预测 """
    model = torch.load(os.path.join(config.save_dir,"medical_ner.ckpt"), 
                       map_location="cpu")
    model.eval()
    
    if not input_str:
        input_str = input("请输入文本: ")    
    
    _, char_ids, len_ids, _ = prepare_dataset([input_str], char_to_id, tag_to_id, pred=True)[0]
    char_tensor = torch.LongTensor(char_ids).view(1,-1)
    len_tensor = torch.LongTensor(len_ids).view(1,-1)
    
    with torch.no_grad():
        
        """ 得到维特比解码后的路径，并转换为标签 """
        paths = model(char_tensor,len_tensor)    
        tags = [id_to_tag[idx] for idx in paths[0]]
    
    pprint(result_to_json(input_str, tags))

#%%
if __name__ == "__main__":
    #cur_dir_path = os.path.abspath(__file__)
    #os.chdir(cur_dir_path)
    if config.train:
        train()
    
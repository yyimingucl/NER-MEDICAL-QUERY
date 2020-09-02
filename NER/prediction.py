# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 09:26:40 2020

@author: yyimi

Predicition 
"""
import os
import sys
sys.path.append('..')

from NER_data import prepare_dataset,load_word2id,load_tag2id
from NER_parameters import params
from NER_functions import get_entity
import torch


#%%
config = params()

def predict(input_str):
    
    char_to_id, id_to_char = load_word2id(config.word2id)
    tag_to_id, id_to_tag = load_tag2id(config.tag2id)
    
    """ 用cpu预测 """
    model = torch.load(os.path.join(config.save_dir,"medical_ner.ckpt"), 
                       map_location="cpu")
    model.eval()
    
    #if not input_str:
    #    input_str = input("请输入文本: ")    
    
    _, char_ids, len_ids, _ = prepare_dataset([input_str], char_to_id, tag_to_id, pred=True)[0]
    char_tensor = torch.LongTensor(char_ids).view(1,-1)
    len_tensor = torch.LongTensor(len_ids).view(1,-1)
        
    char_seq = []
    for char in input_str:
        char_seq.append(char)
        
    with torch.no_grad():
        
        """ 得到维特比解码后的路径，并转换为标签 """
        paths = model(char_tensor,len_tensor)    
        tags = [id_to_tag[idx] for idx in paths[0]]
        
    PATIENT,PART,SIGN,DISEASE,TREATMENT,QUESTION = get_entity(char_seq,tags)
    
    res = {'PATIENT':PATIENT,'PART':PART,'SIGN':SIGN,
           'DISEASE':DISEASE,'TREATMENT':TREATMENT,'QUESTION':QUESTION}
    print(res)
    return res 

#%%
    
if __name__ == '__main__':
    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.chdir(parent_path)
    
    input_str = input("请输入文本: ")
    res = predict(input_str)
    
    
    
    
    
    

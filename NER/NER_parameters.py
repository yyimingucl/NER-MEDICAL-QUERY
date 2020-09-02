# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:01:08 2020

@author: yyimi

设置各类参数
"""

import argparse
import numpy as np
import random
import torch
import os

from NER_functions import str2bool
#%%
def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_manual_seed(50)
print("设置随机数种子为50")


root_path = os.path.dirname(os.path.abspath(__file__))

def params():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    # for model file
    add_arg("--len_dim",default=20, help="Embedding size for length, 0 if not used", 
            type=int)
    add_arg("--char_dim", default=100, help="Embedding size for characters", 
            type=int)
    add_arg("--hidden_dim", default=256, help="Num of hidden units in LSTM", 
            type=int)
    add_arg("--dropout", default=0.5, help="Dropout rate", type=float)
    
    
    
    
    # for data file
    add_arg("--train_file", default=os.path.join(root_path+"//data","train_data.csv"),
            help="file for train data",type=str)   
    add_arg("--dev_file", default=os.path.join(root_path+"//data","dev_data.csv"),
            help="file for dev data",type=str)
    add_arg("--test_file", default=os.path.join(root_path+"//data","test_data.csv"),
            help="file for test data",type=str)
    add_arg("--word2id", default=os.path.join(root_path+"//data","word2id.csv"),
            help="file for word2id mapping",type=str)
    add_arg("--pre_embedding", default = os.path.join(root_path+"//resource","pre_embedding.csv"),
            help="file for pre_embedding char vector", type=str)
    add_arg("--external_vocab", default = os.path.join(root_path+"//resource","medical_vocab.csv"),
            help="file for external vocab", type=str)
    add_arg("--tag2id", default = os.path.join(root_path+"//data","tag2id.csv"),
            help="file for tag2id mapping", type=str)
    
    
    
    #for main file
    add_arg("--batch_size", default = 10, help="batch size", type=int)
    add_arg("--lr", default=0.005, help="Initial learning rate", type=float)
    add_arg("--weight_decay", default=1e-5, help="Learning rate decay", type=float)
    add_arg("--optimizer", default="adam", help="Optimizer for training",type=str)
    add_arg("--max_epoch", default=50, help="maximum training epochs",type=int)
    add_arg("--clip", default=5, help="Gradient clip", type=float)
    add_arg("--steps_check", default=100, help="steps per checkpoint",type=int)
    add_arg("--save_dir", default=os.path.join(root_path,"result"), 
            help="Path to save model",type=str)
    add_arg("--require_improve", default=5000, help="Max step for early stop",
            type=int)
    add_arg("--model_type", default="bilstm", help="Model type, can be idcnn or bilstm"
            ,type=str)
   
    
    #for log file
    add_arg("--log_file", default="train.log", help="File for log",type=str)
    add_arg("--train", default=True, help="Whether train the model",type=str2bool)
    
    args = parser.parse_args()
    return args
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

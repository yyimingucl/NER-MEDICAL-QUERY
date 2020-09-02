# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 01:59:46 2020

@author: yyimi

将文本数据词袋化，统计词频，建立词典
"""

import pymysql
import re
import jieba
import csv
from itertools import chain 
import os 

#%% 数据处理
def clean_data(data, n = 5):
    '''
    Parameters
    ----------
    data : list 
        待清洗的数据
    n : int
        最小文本长度
    Returns
    -------
    去除特殊符号的文本信息，筛选有用信息
    '''
    clean = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    remove_num = 0 
    for i in range(len(data)):
        j = i - remove_num
        temp = clean.sub('',data[j][0])
        if len(temp) <= n :
            data.pop(j)
            remove_num+=1
        else:
            data[j] = temp.strip()
        
    return list(set(data)) #去重


#%% Stop word&Cutword
def load_stop_word(path):
    '''
    Parameters
    ----------
    path : 路径
        停用词路径(目录）
    Returns
    -------
    停用词列表
    '''
    with open(path+'\\resource\\stop_word.csv',mode = 'r',encoding = 'utf8') as F:
        row = csv.reader(F, delimiter=',')
        stopword_list = []
        for i in row:
            stopword_list.append(i[0])
    return stopword_list

def cut_word(sentence):
    '''
    Parameters
    ----------
    sentence : string
        文本信息.
    Returns
    -------
    分词结果
    '''
    res = []
    for word in jieba.cut(sentence):
        for char in word:
            if char.isdigit():
                char = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                char = '<ENG>'
            res.append(char)
    #char_level process
        
    return list(chain(res))
      
#%%统计词频(切词后）建立词典
from collections import Counter
def word_to_id(word_list,path,top=5000):
    '''
    Parameters
    ----------
    word_list : list
        切词结果
    词袋化(5000个最高频词)
    Returns
    -------
    sortlist : dict
    '''
    word_list = dict(Counter(word_list))
    word_list = sorted(word_list.items(), key=lambda f:f[1], reverse = True)
    
    word2id = {('<PAD>',0):0,('<UNK>',1):1}
    max_num = min(top,len(word_list))
    for index,word in enumerate(word_list[:max_num]):
        word2id[word[0]]=index+2
        
    with open(path+'\\data\\word2id.csv', 'w', encoding = 'utf-8') as f:
        [f.write('{0},{1}\n'.format(key[0], value)) for key, value in word2id.items()]
        
    return word2id

#%% Define tag2id file 
def tag_to_id(path):
    tag2id = {'<PAD>':0,'O':1,
        'B-PATIENT':2, 'I-PATIENT':3,
        'B-PART':4, 'I-PART':5,
        'B-SIGN':6,'I-SIGN':7,
        'B-DISEASE':8,'I-DISEASE':9,
        'B-TREATMENT':10, 'I-TREATMENT':11,
        'B-QUESTION':12,'I-QUESTION':13}
    
    with open(path+'\\data\\tag2id.csv', 'w', encoding = 'utf-8') as f:
        [f.write('{0},{1}\n'.format(key, value)) for key, value in tag2id.items()]
    
    return tag2id

#%%
if __name__ == "__main__" :
    db = pymysql.connect("127.0.0.1", "root", "YYMabc990906",
                         "q_a", charset='utf8')
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    cursor.execute("SELECT question FROM t_adv_order")
    Dataset_Q = cursor.fetchall()
    db.close()
    txt_data = clean_data(list(Dataset_Q))
    
    
    current_work_dir = os.path.dirname(os.path.abspath(__file__))
    stopword = load_stop_word(current_work_dir)
    
    jieba.load_userdict(current_work_dir+'\\resource\\medical_vocab.csv')
    txts = []
    for case in txt_data:
        for word in cut_word(case):
            if word not in stopword:
                txts.append(word) 

    word2id = word_to_id(txts, current_work_dir)
    tag2id = tag_to_id(current_work_dir) 








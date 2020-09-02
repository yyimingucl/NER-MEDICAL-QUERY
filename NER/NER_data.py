# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:10:51 2020

@author: yyimi

加载 & 标准化模型数据
"""

import numpy as np
import codecs
#import torch
from NER_functions import get_length_feature,jieba
from NER_parameters import params

config = params()

#%%建立医疗实体和数据标准的对应关系 BIO标注
tag2id = {'<PAD>':0,'O':1,
        'B-PATIENT':2, 'I-PATIETNT':3,
        'B-PART':4, 'I-PART':5,
        'B-SIGN':6,'I-SIGN':7,
        'B-DISEASE':8,'I-DISEASE':9,
        'B-TREATMENT':10, 'I-TREATMENT':11,
        'B-QUESTION':12,'I-QUESTION':13}


#%%Embedding
def build_character_embeddings(pretrained_emb_path, 
                               word2id, embedding_dim):
    print('loading pretrained embeddings from {}'.format(pretrained_emb_path))
    pre_emb = {}
    for line in codecs.open(pretrained_emb_path, 'r', 'utf-8'):
        line = line.strip().split()
        if len(line) == embedding_dim + 1:
            pre_emb[line[0]] = [float(x) for x in line[1:]]
            
    word_ids = sorted(word2id.items(), key=lambda x: x[1])
    characters = [c[0] for c in word_ids]
    embeddings = list()
    for i, ch in enumerate(characters):
        if ch in pre_emb:
            embeddings.append(pre_emb[ch])
        else:
            embeddings.append(np.random.uniform(-0.25, 0.25, embedding_dim).tolist())
    embeddings = np.asarray(embeddings, dtype=np.float32)
    #np.save(embeddings_path, embeddings)
    return embeddings

#%% Data_loading Process

def load_word2id(path):
    with open(path,mode = 'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        word2id={}
        id2word={}
        for line in lines:
            line = line.split(',')
            word2id[str(line[0])] = int(line[1])        
            id2word[int(line[1])] = str(line[0])
    f.close()
    
    return word2id, id2word

def load_tag2id(path):
    with open(path,mode = 'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        tag2id = {}
        for line in lines:
            line = line.split(',')
            tag2id[str(line[0])] = int(line[1])
            
    f.close()
    id2tag = dict(zip(tag2id.values(), tag2id.keys()))
    return tag2id, id2tag
        
            
        
    
            
def load_sentence(path):
    '''
    Parameters
    ----------
    path : 标记数据的储存目录
    Returns
    -------
    sentences: [[[['无', 'O'], ['长', 'B-Dur'], ['期', 'I-Dur'],...]]
    '''
    sentences = []
    sentence = []
    with open(path, mode = 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != '':
                sentence.append([line[0],line[2:]])
            else:
                sentences.append(sentence)
                sentence = []
    
    return sentences
                
def prepare_dataset(sentences, char2id, tag2id, pred = False):
    '''
    Parameters
    ----------
    sentences : List
        Processed by load_sentence function
    char_to_id : dict
        char2id mapping
    tag_to_id : dict
        tag2id mapping.
    pred : Boolen
        whether used to predict new sentence
    Returns
    -------
    List
    each elements contain (chars_idx, lens_idx, tags_idx) for a sentence
    '''
    data = []
    
    for sen in sentences:
        chars = [w[0] for w in sen]
        tags = [w[-1] for w in sen]
        
        
        chars_idx = [char2id[c if c in char2id else '<UNK>'] for c in chars]
        lens_idx = get_length_feature("".join(chars))
        
        if not pred:
            tags_idx =  [tag2id[t] for t in tags]
        else:
            tags_idx = [tag2id["<PAD>"] for _ in tags]
        
        
        assert len(chars_idx) == len(lens_idx) == len(tags_idx)
        data.append([chars, chars_idx, lens_idx, tags_idx])
    
    return data
        
#%% main function
    
def build_dataset():
    #load sentence 
    train_sentence = load_sentence(config.train_file)
    dev_sentence = load_sentence(config.dev_file)
    test_sentence = load_sentence(config.test_file)
    
    #load various mapping dict 
    word2id, id2word = load_word2id(config.word2id)
    tag2id, id2tag = load_tag2id(config.tag2id)
    
    #load pretrained embedding char vector
    #produce random vector if char is not pretrained
    #used pretrained char vector is in 100 dimension
    emb_matrix = build_character_embeddings(config.pre_embedding, word2id, 
                                            embedding_dim=100) 
    
    
    #set external vocab for jieba
    jieba.load_userdict(config.external_vocab)
    
    
    
    #process data to be used in model
    train_data = prepare_dataset(train_sentence, word2id, tag2id)
    dev_data = prepare_dataset(dev_sentence, word2id, tag2id)
    test_data = prepare_dataset(test_sentence, word2id, tag2id)
    
    
    return train_data,dev_data,test_data,word2id,id2word,tag2id,id2tag,emb_matrix
    
    
    
    
    

    











    

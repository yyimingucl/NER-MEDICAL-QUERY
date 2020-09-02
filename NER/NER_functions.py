# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:40:45 2020

@author: yyimi

待使用的函数
"""

import sys
sys.path.append('..')
import os
import jieba
import argparse
from CONLLeval import return_report


#%%
def get_length_feature(string):
    '''
    the augment of length feature of words on the char vector
    every length feature of a word is an id between 0-3
    (0 for single word, 1 for start, 2 for in, 3 for end)
    Parameters
    ----------
    string： 
        example ['全世界']
    Returns
    -------
        [1,2,3]
    '''
    length_feature = []
    for word in jieba.cut(string):
        if len(word) == 1:
            length_feature.append(0)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            length_feature.extend(tmp)
    return length_feature
    

#%%
def str2bool(v):
    '''
    Justify whether the string means a boolean value
    ------------------------------------------------
    Retur True, False or Raise Error
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def same_ele_index(List,Element):
    '''
    Find the index of all the same elements in the list 
    Parameters
    ----------
    List : list
    Element: int,list,tuple,str....
    Returns
    -------
    List 
    the index of all the elements in list
    '''
    total_num = List.count(Element)
    res = []
    for i in range(len(List)):
        if List[i] == Element:
            res.append(i)
        if len(res)==total_num:
            break
    return res
            

#%%实体抽取
def get_entity(char_seq, tag_seq):
    
    PATIENT = PATIENT_entity(char_seq, tag_seq)
    PART = PART_entity(char_seq, tag_seq)
    SIGN = SIGN_entity(char_seq, tag_seq)
    DISEASE = DISEASE_entity(char_seq, tag_seq)
    TREATMENT = TREATMENT_entity(char_seq, tag_seq)
    QUESTION = QUESTION_entity(char_seq, tag_seq)
    
    return PATIENT, PART, SIGN, DISEASE, TREATMENT, QUESTION


def PATIENT_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    PATIENT = []
    patient = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        
        if tag == 'B-PATIENT':
            patient = char
            if i + 1 == length or tag_seq[i+1] != 'I-PATIENT':
                PATIENT.append(patient)
                
        if tag == 'I-PATIENT':
            patient += char
            if i + 1 == length or tag_seq[i+1] != 'I-PATIENT':
                PATIENT.append(patient)
            
    return PATIENT


def DISEASE_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    DISEASE = []
    disease = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):

        if tag == 'B-DISEASE':
            disease = char
            if i + 1 == length or tag_seq[i+1] != 'I-DISEASE':
                DISEASE.append(disease)
                
        if tag == 'I-DISEASE':
            disease += char
            if i + 1 == length or tag_seq[i+1] != 'I-DISEASE':
                DISEASE.append(disease)
            
    return DISEASE


def PART_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    PART = []
    part = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        
        if tag == 'B-PART':
            part = char
            if i + 1 == length or tag_seq[i+1] != 'I-PART':
                PART.append(part)
                
        if tag == 'I-PART':
            part += char
            if i + 1 == length or tag_seq[i+1] != 'I-PART':
                PART.append(part)
            
    return PART


def SIGN_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    SIGN = []
    sign = ''
   
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        
        if tag == 'B-SIGN':
            sign = char
            if i + 1 == length or tag_seq[i+1] != 'I-SIGN':
                SIGN.append(sign)
                
        if tag == 'I-SIGN':
            sign += char
            if i + 1 == length or tag_seq[i+1] != 'I-SIGN':
                SIGN.append(sign)
            
    return SIGN

  
def TREATMENT_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    TREATMENT = []
    treatment = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        
        if tag == 'B-TREATMENT':
            treatment = char
            if i + 1 == length or tag_seq[i+1] != 'I-TREATMENT':
                TREATMENT.append(treatment)
                
        if tag == 'I-TREATMENT':
            treatment += char
            if i + 1 == length or tag_seq[i+1] != 'I-TREATMENT':
                TREATMENT.append(treatment)
            
    return TREATMENT
    
    
def QUESTION_entity(char_seq, tag_seq):
    '''
    Parameters
    ----------
    tag_seq : List
        the tag on each character
    char_seq : List
        characters list
    Returns
    -------
    PATIENT : list
        the entity in the word
    '''
    length = len(char_seq)
    QUESTION = []
    question = ''
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        
        if tag == 'B-QUESTION':
            question = char
            if i + 1 == length or tag_seq[i+1] != 'I-QUESTION':
                QUESTION.append(question)
                
        if tag == 'I-QUESTION':
            question += char
            if i + 1 == length or tag_seq[i+1] != 'I-QUESTION':
                QUESTION.append(question)
            
    return QUESTION




#%%
def test_ner(results, path):
    """
    用CoNLL-2000的实体识别评估脚本来评估模型
    """
    
    """ 用CoNLL-2000的脚本，需要把预测结果保存为文件，再读取 """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def result_to_json(string, tags):
    """ 按规范的格式输出预测结果 """
    
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

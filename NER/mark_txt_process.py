# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 07:31:39 2020

@author: yyimi

文本标注处理
"""

import json,os
from NER_functions import same_ele_index

#%%
def read_corpus(path):
    '''
    Parameters
    ----------
    path : 
        A file processed by doccano
    Returns
    -------
    result : list

    '''
    result = []
    with open(path,mode='r',encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            result.append(json.loads(line))
    f.close()
    return result


#%%
def label_process(doccano_file,store_path):
    '''
    Parameters
    ----------
    doccano_file : list
        A list includes dicts which contain the 
        labelling information of each text
    store_path : TYPE
        The location to store the labelled file 
    Returns
    -------
    labelled_txt : List
        contains tuples (char,tag)

    '''
    labelled_txt = []
    for each in doccano_file:
        if each['labels'] == []:
            continue
        else:
            i = 0 
            for index,char in enumerate(each['text']):
                if index == each['labels'][i][0]:
                    labelled_txt.append((char,'B-'+ each['labels'][i][2]))
                    #consider one char entity
                    if index == each['labels'][i][1]-1:
                        i+=1
                        i = min(len(each['labels'])-1, i)
                        
                elif index <= each['labels'][i][1]-1 and index > each['labels'][i][0]:
                    labelled_txt.append((char,'I-'+ each['labels'][i][2]))
                    if index == each['labels'][i][1]-1:
                        i+=1
                        i = min(len(each['labels'])-1, i)
                else:
                    labelled_txt.append((char,'O'))
        labelled_txt.append('')
    
    
    split_sentence_loc = same_ele_index(labelled_txt, '')
    
    loc1 = int(labelled_txt.count('')*0.6+1)
    loc2 = loc1+int(0.2*labelled_txt.count('')+1)   
    train_part = split_sentence_loc[loc1]
    dev_part = split_sentence_loc[loc2]
    
    
    
    train_data = labelled_txt[:train_part]
    dev_data = labelled_txt[train_part:dev_part]
    test_data = labelled_txt[dev_part:]
    
    with open(store_path+'\\train_data.csv', 'w',encoding = 'utf-8') as f:
        for each in train_data:
            if each != '':
                f.write('{0},{1}\n'.format(each[0], each[1]))
            else:
                f.write('\n')
    
    with open(store_path+'\\dev_data.csv', 'w',encoding = 'utf-8') as f:
        for each in dev_data:
            if each != '':
                f.write('{0},{1}\n'.format(each[0], each[1]))
            else:
                f.write('\n')
    
    with open(store_path+'\\test_data.csv', 'w',encoding = 'utf-8') as f:
        for each in test_data:
            if each != '':
                f.write('{0},{1}\n'.format(each[0], each[1]))
            else:
                f.write('\n')
    
    return labelled_txt
#%%                  


if __name__ == '__main__':
    
    cur_dir_name = current_work_dir = os.path.dirname(os.path.abspath(__file__))
    label_process(read_corpus(cur_dir_name+'\\resource\\label.json1'),
                  cur_dir_name+'\\data')
    






               
                
                
            

        
             
        
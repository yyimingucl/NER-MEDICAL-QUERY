# NER-MEDICAL-QUERY
BILSTM+CRF NER by pytorch
main structure from (https://github.com/DengYangyong/medical_entity_recognize)

## 1. environments
python==3.7

torch==1.6.0

jieba==0.42.1

## 2.Mark the Sample Data
around 10000 pieces queries from donghuayiwei-jiankangle online medical query platform
Marked by Doccano (https://github.com/doccano/doccano)

## 3.Model Evaulation
There are severe problem at marking data, and the resulting perfomance on dev_set is quite bad (F1-0.565)

## 4.Layout
model folder store the main structure and CRF layer

sql_file store the original sql file (manipulate by pymysql in NER_data)

Batch.py: batch the train sample with similar length of words

build_vocab.py: bagging the characters of train_sample

predict.py: use the model to predict new inputs

### command line performance
![image]()

main: train the model

mark_txt_process: transfer the marked queries produced by doccano to standard training sample

### data from doccano 
![image]()

### standard training sample
![image]()

NER_data: prepare and clean data

NER_functions: Used functions

NER_parameters: Used parameters

## 5.Prediction
use the prediction.py in results folder



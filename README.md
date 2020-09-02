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
### model folder store the main structure and CRF layer
### sql_file store the original sql file (manipulate by pymysql in NER_data)




import json
import os
import argparse
import warnings
from tqdm import tqdm
import numpy
#--------------------------------------------------------------#
#----------------获取AGQA中无法检测到任何对象的图像列表id-----------#
#--------------------------------------------------------------#
train_dir='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/train/'
test_dir='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/test/'

train=os.listdir(train_dir)
test=os.listdir(test_dir)

bad_detections_train=[]
bad_detections_test=[]
bad_detections_val=[]

for name in train:
    path=os.path.join(train_dir,name)
    file=open(path,'r')
    detection=json.load(file)
    if len(detection['detections'])==0:
        bad_detections_train.append(detection['image_name'])
    else:
        flag=[]
        for i in range(len(detection['detections'])):
            flag.append(detection['detections'][i]['class_no'])
        if 1 not in flag:
            bad_detections_train.append(detection['image_name'])

for name in test:
    path=os.path.join(test_dir,name)
    file=open(path,'r')
    detection=json.load(file)
    if len(detection['detections'])==0:
        bad_detections_test.append(detection['image_name'])
        bad_detections_val.append(detection['image_name'])
    else:
        flag=[]
        for i in range(len(detection['detections'])):
            flag.append(detection['detections'][i]['class_no'])
        if 1 not in flag:
            bad_detections_test.append(detection['image_name'])
            bad_detections_val.append(detection['image_name'])

bad=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/bad.json','w')
write= {'bad_detections_train': bad_detections_train,
        'bad_detections_test' : bad_detections_test,
        'bad_detections_val' : bad_detections_val}
json.dump(write, bad, ensure_ascii=False, sort_keys=True, indent=4)


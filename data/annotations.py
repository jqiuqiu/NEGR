import json
import os.path
import pickle

import numpy
from tqdm import tqdm
#--------------------------------------------------------------#
#---------------------获取AGQA的Annotations---------------#
#--------------------------------------------------------------#
path='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG'
object_file=open(os.path.join(path,'object_bbox_and_relationship.pkl'),'rb')
person_file=open(os.path.join(path,'person_bbox.pkl'),'rb')
train_file=open(os.path.join(path,'train_annotations.json'),'w')
test_file=open(os.path.join(path,'test_annotations.json'),'w')

object_datas=pickle.load(object_file,encoding='latin1')
person_datas=pickle.load(person_file,encoding='latin1')

train=os.listdir('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/train_hierarchies')
for i in range(len(train)):
    train[i]=train[i].replace('.json','')

test=os.listdir('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/test_hierarchies')
for i in range(len(test)):
    test[i]=test[i].replace('.json','')

train_datas={}
test_datas={}
pbar = tqdm(total=len(object_datas))
for object_data in object_datas:
    split=object_data.split('/')[0].replace('.mp4','')
    if split in train:
        person=person_datas[object_data]
        train_datas[object_data] = []
        for i in range(len(object_datas[object_data])):
            if object_datas[object_data][i]['visible']==False:
                data = {}
                if person['bbox'].shape==(0,4):
                    data['person_bbx'] = numpy.zeros(shape=(1,4)).tolist()[0]
                else:
                    data['person_bbx'] = person['bbox'].tolist()[0]
                data['Verbs'] = None
                data['object']= {"obj_bbx":[]}
                train_datas[object_data].append(data)

            else:
                for j in range(len(object_datas[object_data][i]['contacting_relationship'])):
                    data = {}
                    if person['bbox'].shape == (0, 4):
                        data['person_bbx'] = numpy.zeros(shape=(1, 4)).tolist()[0]
                    else:
                        data['person_bbx'] = person['bbox'].tolist()[0]
                    data['Verbs']=object_datas[object_data][i]['contacting_relationship'][j]
                    data['object']={"obj_bbx":[object_datas[object_data][i]['bbox']]}
                    train_datas[object_data].append(data)
    if split in test:
        person=person_datas[object_data]
        test_datas[object_data] = []
        for i in range(len(object_datas[object_data])):
            if object_datas[object_data][i]['visible']==False:
                data = {}
                if person['bbox'].shape == (0, 4):
                    data['person_bbx'] = numpy.zeros(shape=(1, 4)).tolist()[0]
                else:
                    data['person_bbx'] = person['bbox'].tolist()[0]
                data['Verbs'] = None
                data['object']= {"obj_bbx":[]}
                test_datas[object_data].append(data)

            else:
                for j in range(len(object_datas[object_data][i]['contacting_relationship'])):
                    data = {}
                    if person['bbox'].shape == (0, 4):
                        data['person_bbx'] = numpy.zeros(shape=(1, 4)).tolist()[0]
                    else:
                        data['person_bbx'] = person['bbox'].tolist()[0]
                    data['Verbs']=object_datas[object_data][i]['contacting_relationship'][j]
                    data['object']={"obj_bbx":[object_datas[object_data][i]['bbox']]}
                    test_datas[object_data].append(data)
    pbar.update(1)
pbar.close()

json.dump(train_datas, train_file, ensure_ascii=False, sort_keys=True, indent=4)
json.dump(test_datas, test_file, ensure_ascii=False, sort_keys=True, indent=4)

import os
import pickle
from tqdm import tqdm
import json
#--------------------------------------------------------------#
#---------------------获取AGQA的Object_Detections---------------#
#--------------------------------------------------------------#
path='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG'
object_file=open(os.path.join(path,'object_bbox_and_relationship.pkl'),'rb')
person_file=open(os.path.join(path,'person_bbox.pkl'),'rb')
train_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_train_stsgs.pkl','rb'))
test_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_test_stsgs.pkl','rb'))

train=train_sgg_file.keys()
test=test_sgg_file.keys()
'''train=['00HFP','00MFE','00N38']
test=['00T1E','0A8CF']'''

object_datas=pickle.load(object_file,encoding='latin1')
person_datas=pickle.load(person_file,encoding='latin1')

object_index=['person','bag','bed','blanket','book','box','broom','chair','closet/cabinet',
              'clothes','cup/glass/bottle','dish','door','doorknob','doorway','floor','food',
              'groceries','laptop','light','medicine','mirror','paper/notebook','phone/camera',
              'picture','pillow','refrigerator','sandwich','shelf','shoe','sofa/couch','table',
              'television','towel','vacuum','window']


pbar = tqdm(total=len(object_datas))
for object_data in object_datas:
    name=object_data.replace('/','_')
    mp4=name.split('_')[0].replace('.mp4','')
    if mp4 in train:
        train_file=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/train/{}.json'.format(name),'w')
    elif mp4 in test:
        test_file=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/test/{}.json'.format(name),'w')
    person=person_datas[object_data]
    data = {}
    data['H']=person['bbox_size'][1]
    data['W']=person['bbox_size'][0]
    data['image_name']=object_data

    data['detections']=[]
    detections = {}
    if len(person['bbox'])!=0:
        detections['class_str'] = 'person'
        detections['score'] = float(person['bbox_score'])
        detections['class_no'] = 1
        detections['box_coords'] = person['bbox'].tolist()[0]
        data['detections'].append(detections)
    for i in range(len(object_datas[object_data])):
        if object_datas[object_data][i]['visible']==False:
            continue
        else:
            detections={}
            detections['class_str']=object_datas[object_data][i]['class']
            detections['score']=0.9
            detections['class_no']=object_index.index(detections['class_str'])+1
            detections['box_coords']=object_datas[object_data][i]['bbox']
            data['detections'].append(detections)
    if mp4 in train:
        json.dump(data, train_file, ensure_ascii=False, sort_keys=True, indent=4)
    elif mp4 in test:
        json.dump(data, test_file, ensure_ascii=False, sort_keys=True, indent=4)

    pbar.update(1)
pbar.close()


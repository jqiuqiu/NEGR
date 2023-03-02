import os
import pickle
from tqdm import tqdm
import numpy
import json
#---------------------------------------------------------------#
#---------------------获取AGQA的整体场景图-------------------------#
#---------------------+-----------------------------------------#

index_file=json.load(open('/home/lll/AGQA-dest/dest/data/ENG.txt','r'))

train_write_file=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/46GP8_SGG.json','w')
test_write_file=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/00T1E_SGG.json','w')
bbox_dir=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/frame_list.txt','r')
img_ids=bbox_dir.readlines()
train_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_train_stsgs.pkl','rb'))
test_sgg_file=pickle.load(open('/home/lll/AGQA-dest/dest/data/AGQA_scene_graphs/AGQA_test_stsgs.pkl','rb'))

#train=train_sgg_file.keys()
#test=test_sgg_file.keys()
train=['46GP8']
test=['00T1E']
all_score=[#verb
    "awakening","closing","cooking","dressing","drinking","fixing","grasping","laughing","lying","making",
    "opening","photographing","playing on","pouring","putting down","running","sitting","smiling","sneezing",
    "snuggling","standing","taking","talking","throwing","tidying","turning","undressing","walking",
    "washing","watching","working on",
	#relatioin
	"looking at","not looking at","unsure","above","beneath","in front of","behind","on the side of","in",
	"carrying","covered by","drinking from","eating","having it on the back","holding","leaning on","lying on",
	"not contacting","other relationship","sitting on","standing on","touching","twisting","wearing","wiping",
    "writing on"]

train_sgg_info = {}
test_sgg_info = {}
pbar = tqdm(total=len(img_ids))
for img_id in img_ids:
    mp4=img_id.split('/')[0].replace('.mp4','')
    img=img_id.split('/')[1].replace('.png\n','')
    img_id=img_id.replace('\n','')
    img_path=img_id.replace('/','_').replace('\n','')+'.json'
    if mp4 in train:
        bbox_file=json.load(open(os.path.join('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/train/',img_path),'r'))
    elif mp4 in test:
        bbox_file = json.load(open(os.path.join('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/test/', img_path),'r'))
    info={}
    info['bbox']=[]
    info['bbox_labels']=[]
    info['bbox_scores']=[]

    detections=bbox_file['detections']
    for i in range(len(detections)):
        info['bbox'].append(detections[i]['box_coords'])
        info['bbox_labels'].append(detections[i]['class_str'])
        info['bbox_scores'].append(detections[i]['score'])

    info['rel_attention_pairs'] = [[]] * len(detections)
    info['rel_attention_labels'] = [[]] * len(detections)
    info['rel_attention_scores'] = [[]] * len(detections)
    info['rel_attention_all_scores'] = [[0.1] * len(all_score)] * len(detections)
    info['rel_contact_pairs'] = [[]] * len(detections)
    info['rel_contact_labels'] = [[]] * len(detections)
    info['rel_contact_scores'] = [[]] * len(detections)
    info['rel_contact_all_scores'] = [[0.1] * len(all_score)] * len(detections)
    info['rel_spatial_pairs'] = [[]] * len(detections)
    info['rel_spatial_labels'] = [[]] * len(detections)
    info['rel_spatial_scores'] = [[]] * len(detections)
    info['rel_spatial_all_scores'] = [[0.1] * len(all_score)] * len(detections)
    info['rel_verb_pairs'] = [[]] * len(detections)
    info['rel_verb_labels'] = [[]] * len(detections)
    info['rel_verb_scores'] = [[]] * len(detections)
    info['rel_verb_all_scores'] = [[0.1] * len(all_score)] * len(detections)

    if len(detections)!=0:
        if 'person' in info['bbox_labels']:
            person_index=info['bbox_labels'].index('person')
        else:
            person_index=len(info['bbox_labels'])
            info['bbox'].append(numpy.zeros(shape=(1,4)).tolist()[0])
            info['bbox_labels'].append('person')
            info['bbox_scores'].append(0.9)

        info['rel_attention_pairs'] = [None] * len(info['bbox_labels'])
        info['rel_attention_labels'] = [None] * len(info['bbox_labels'])
        info['rel_attention_scores'] = [None] * len(info['bbox_labels'])
        info['rel_attention_all_scores'] = [[0.1] * len(all_score)] * len(info['bbox_labels'])
        info['rel_contact_pairs'] = [None] * len(info['bbox_labels'])
        info['rel_contact_labels'] = [None] * len(info['bbox_labels'])
        info['rel_contact_scores'] = [None] * len(info['bbox_labels'])
        info['rel_contact_all_scores'] = [[0.1] * len(all_score)] * len(info['bbox_labels'])
        info['rel_spatial_pairs'] = [None] * len(info['bbox_labels'])
        info['rel_spatial_labels'] = [None] * len(info['bbox_labels'])
        info['rel_spatial_scores'] = [None] * len(info['bbox_labels'])
        info['rel_spatial_all_scores'] = [[0.1] * len(all_score)] * len(info['bbox_labels'])
        info['rel_verb_pairs'] = [None] * len(info['bbox_labels'])
        info['rel_verb_labels'] = [None] * len(info['bbox_labels'])
        info['rel_verb_scores'] = [None] * len(info['bbox_labels'])
        info['rel_verb_all_scores'] = [[0.1] * len(all_score)] * len(info['bbox_labels'])

        for i in range(len(info['bbox_labels'])):
            info['rel_attention_labels'][i] = []
            info['rel_attention_scores'][i] = []
            info['rel_attention_pairs'][i] = []
            info['rel_contact_labels'][i] = []
            info['rel_contact_scores'][i] = []
            info['rel_contact_pairs'][i] = []
            info['rel_spatial_labels'][i] = []
            info['rel_spatial_scores'][i] = []
            info['rel_spatial_pairs'][i] = []
            info['rel_verb_labels'][i] = []
            info['rel_verb_scores'][i] = []
            info['rel_verb_pairs'][i] = []

        if mp4 in train:
            for i in range(len(train_sgg_file[mp4][img]['attention']['vertices'])):
                rel_attention_name=train_sgg_file[mp4][img]['attention']['names'][i]
                rel_attention=index_file[rel_attention_name]
                for j in range(len(train_sgg_file[mp4][img]['attention']['vertices'][i]['objects'])):
                    rel_attention_object_class=train_sgg_file[mp4][img]['attention']['vertices'][i]['objects'][j]['class']
                    rel_attention_object=index_file[rel_attention_object_class]
                    if rel_attention_object in info['bbox_labels']:
                        obj_index=info['bbox_labels'].index(rel_attention_object)
                        info['rel_attention_labels'][obj_index].append(rel_attention)
                        info['rel_attention_scores'][obj_index].append(0.9)
                        info['rel_attention_pairs'][obj_index].append([person_index,obj_index])
                        info['rel_attention_all_scores'][obj_index][all_score.index(rel_attention)]=0.9

            for i in range(len(info['bbox_labels'])):
                if info['rel_attention_labels'][i]==[]:
                    info['rel_attention_scores'][i].append(0.1)
                    info['rel_attention_pairs'][i].append([-1,-1])
                    info['rel_attention_all_scores'][i]=[0.1]*len(all_score)


            for i in range(len(train_sgg_file[mp4][img]['contact']['vertices'])):
                rel_contact_name=train_sgg_file[mp4][img]['contact']['names'][i]
                rel_contact=index_file[rel_contact_name]
                for j in range(len(train_sgg_file[mp4][img]['contact']['vertices'][i]['objects'])):
                    rel_contact_object_class=train_sgg_file[mp4][img]['contact']['vertices'][i]['objects'][j]['class']
                    rel_contact_object=index_file[rel_contact_object_class]
                    if rel_contact_object in info['bbox_labels']:
                        obj_index=info['bbox_labels'].index(rel_contact_object)
                        info['rel_contact_labels'][obj_index].append(rel_contact)
                        info['rel_contact_scores'][obj_index].append(0.9)
                        info['rel_contact_pairs'][obj_index].append([person_index,obj_index])
                        info['rel_contact_all_scores'][obj_index][all_score.index(rel_contact)]=0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_contact_labels'][i]:
                    info['rel_contact_scores'][i].append(0.1)
                    info['rel_contact_pairs'][i].append([-1,-1])
                    info['rel_contact_all_scores'][i]=[0.1]*len(all_score)


            for i in range(len(train_sgg_file[mp4][img]['spatial']['vertices'])):
                rel_spatial_name=train_sgg_file[mp4][img]['spatial']['names'][i]
                rel_spatial=index_file[rel_spatial_name]
                for j in range(len(train_sgg_file[mp4][img]['spatial']['vertices'][i]['objects'])):
                    rel_spatial_object_class=train_sgg_file[mp4][img]['spatial']['vertices'][i]['objects'][j]['class']
                    rel_spatial_object=index_file[rel_spatial_object_class]
                    if rel_spatial_object in info['bbox_labels']:
                        obj_index=info['bbox_labels'].index(rel_spatial_object)
                        info['rel_spatial_labels'][obj_index].append(rel_spatial)
                        info['rel_spatial_scores'][obj_index].append(0.9)
                        info['rel_spatial_pairs'][obj_index].append([person_index,obj_index])
                        info['rel_spatial_all_scores'][obj_index][all_score.index(rel_spatial)]=0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_spatial_labels'][i]:
                    info['rel_spatial_scores'][i].append(0.1)
                    info['rel_spatial_pairs'][i].append([-1,-1])
                    info['rel_spatial_all_scores'][i]=[0.1]*len(all_score)


            for i in range(len(train_sgg_file[mp4][img]['verb']['vertices'])):
                rel_verb_name=train_sgg_file[mp4][img]['verb']['names'][i]
                rel_verb=index_file[rel_verb_name]
                for j in range(len(train_sgg_file[mp4][img]['verb']['vertices'][i]['objects'])):
                    rel_verb_object_class=train_sgg_file[mp4][img]['verb']['vertices'][i]['objects'][j]['class']
                    rel_verb_object=index_file[rel_verb_object_class]
                    if rel_verb_object in info['bbox_labels']:
                        obj_index=info['bbox_labels'].index(rel_verb_object)
                        info['rel_verb_labels'][obj_index].append(rel_verb)
                        info['rel_verb_scores'][obj_index].append(0.9)
                        info['rel_verb_pairs'][obj_index].append([person_index,obj_index])
                        info['rel_verb_all_scores'][obj_index][all_score.index(rel_verb)]=0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_verb_labels'][i]:
                    info['rel_verb_scores'][i].append(0.1)
                    info['rel_verb_pairs'][i].append([-1,-1])
                    info['rel_verb_all_scores'][i]=[0.1]*len(all_score)

        elif mp4 in test:
            for i in range(len(test_sgg_file[mp4][img]['attention']['vertices'])):
                rel_attention_name = test_sgg_file[mp4][img]['attention']['names'][i]
                rel_attention = index_file[rel_attention_name]
                for j in range(len(test_sgg_file[mp4][img]['attention']['vertices'][i]['objects'])):
                    rel_attention_object_class = test_sgg_file[mp4][img]['attention']['vertices'][i]['objects'][j]['class']
                    rel_attention_object = index_file[rel_attention_object_class]
                    if rel_attention_object in info['bbox_labels']:
                        obj_index = info['bbox_labels'].index(rel_attention_object)
                        info['rel_attention_labels'][obj_index] .append( rel_attention)
                        info['rel_attention_scores'][obj_index] .append( 0.9)
                        info['rel_attention_pairs'][obj_index] .append( [person_index, obj_index])
                        info['rel_attention_all_scores'][obj_index][all_score.index(rel_attention)] = 0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_attention_labels'][i]:
                    info['rel_attention_scores'][i] .append( 0.1)
                    info['rel_attention_pairs'][i] .append( [-1, -1])
                    info['rel_attention_all_scores'][i] = [0.1] * len(all_score)

            for i in range(len(test_sgg_file[mp4][img]['contact']['vertices'])):
                rel_contact_name = test_sgg_file[mp4][img]['contact']['names'][i]
                rel_contact = index_file[rel_contact_name]
                for j in range(len(test_sgg_file[mp4][img]['contact']['vertices'][i]['objects'])):
                    rel_contact_object_class = test_sgg_file[mp4][img]['contact']['vertices'][i]['objects'][j]['class']
                    rel_contact_object = index_file[rel_contact_object_class]
                    if rel_contact_object in info['bbox_labels']:
                        obj_index = info['bbox_labels'].index(rel_contact_object)
                        info['rel_contact_labels'][obj_index] .append( rel_contact)
                        info['rel_contact_scores'][obj_index] .append( 0.9)
                        info['rel_contact_pairs'][obj_index] .append( [person_index, obj_index])
                        info['rel_contact_all_scores'][obj_index][all_score.index(rel_contact)] = 0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_contact_labels'][i]:
                    info['rel_contact_scores'][i] .append( 0.1)
                    info['rel_contact_pairs'][i] .append( [-1, -1])
                    info['rel_contact_all_scores'][i] = [0.1] * len(all_score)

            for i in range(len(test_sgg_file[mp4][img]['spatial']['vertices'])):
                rel_spatial_name = test_sgg_file[mp4][img]['spatial']['names'][i]
                rel_spatial = index_file[rel_spatial_name]
                for j in range(len(test_sgg_file[mp4][img]['spatial']['vertices'][i]['objects'])):
                    rel_spatial_object_class = test_sgg_file[mp4][img]['spatial']['vertices'][i]['objects'][j]['class']
                    rel_spatial_object = index_file[rel_spatial_object_class]
                    if rel_spatial_object in info['bbox_labels']:
                        obj_index = info['bbox_labels'].index(rel_spatial_object)
                        info['rel_spatial_labels'][obj_index] .append( rel_spatial)
                        info['rel_spatial_scores'][obj_index] .append( 0.9)
                        info['rel_spatial_pairs'][obj_index] .append( [person_index, obj_index])
                        info['rel_spatial_all_scores'][obj_index][all_score.index(rel_spatial)] = 0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_spatial_labels'][i]:
                    info['rel_spatial_scores'][i] .append( 0.1)
                    info['rel_spatial_pairs'][i] .append( [-1, -1])
                    info['rel_spatial_all_scores'][i] = [0.1] * len(all_score)

            for i in range(len(test_sgg_file[mp4][img]['verb']['vertices'])):
                rel_verb_name = test_sgg_file[mp4][img]['verb']['names'][i]
                rel_verb = index_file[rel_verb_name]
                for j in range(len(test_sgg_file[mp4][img]['verb']['vertices'][i]['objects'])):
                    rel_verb_object_class = test_sgg_file[mp4][img]['verb']['vertices'][i]['objects'][j]['class']
                    rel_verb_object = index_file[rel_verb_object_class]
                    if rel_verb_object in info['bbox_labels']:
                        obj_index = info['bbox_labels'].index(rel_verb_object)
                        info['rel_verb_labels'][obj_index] .append( rel_verb)
                        info['rel_verb_scores'][obj_index] .append( 0.9)
                        info['rel_verb_pairs'][obj_index] .append( [person_index, obj_index])
                        info['rel_verb_all_scores'][obj_index][all_score.index(rel_verb)] = 0.9

            for i in range(len(info['bbox_labels'])):
                if not info['rel_verb_labels'][i]:
                    info['rel_verb_scores'][i] .append( 0.1)
                    info['rel_verb_pairs'][i] .append( [-1, -1])
                    info['rel_verb_all_scores'][i] = [0.1] * len(all_score)


    if mp4 in train:
        train_sgg_info[img_id]=info
    elif mp4 in test:
        test_sgg_info[img_id]=info

    pbar.update(1)

pbar.close()

json.dump(train_sgg_info, train_write_file, ensure_ascii=False, sort_keys=True, indent=4)
json.dump(test_sgg_info, test_write_file, ensure_ascii=False, sort_keys=True, indent=4)
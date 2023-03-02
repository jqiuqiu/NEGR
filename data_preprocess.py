import json
import os.path

import numpy as np
import torch

#-----------------------------------verb对应id需要修改
VERB2ID={"awakening":0,  #verb
           "closing":1,
           "cooking":2,
           "dressing":3,
           "drinking":4,
           "fixing":5,
           "grasping":6,
           "laughing":7,
           "lying":8,
           "making":9,
           "opening":10,
           "photographing":11,
           "playing on":12,
           "pouring":13,
           "putting down":14,
           "running":15,
           "sitting":16,
           "smiling":17,
           "sneezing":18,
           "snuggling":19,
           "standing":20,
           "taking":21,
           "talking":22,
           "throwing":23,
           "tidying":24,
           "turning":25,
           "undressing":26,
           "walking":27,
           "washing":28,
           "watching":29,
           "working on":30,
	        #relatioin
           "looking at":31,
           "not looking at":32,
           "unsure":33,
           "above":34,
           "beneath":35,
           "in front of":36,
           "behind":37,
           "on the side of":38,
           "in":39,
           "carrying":40,
           "covered by":41,
           "drinking from":42,
           "eating":43,
           "having it on the back":44,
           "holding":45,
           "leaning on":46,
           "lying on":47,
           "not contacting":48,
           "other relationship":49,
           "sitting on":50,
           "standing on":51,
           "touching":52,
           "twisting":53,
           "wearing":54,
           "wiping":55,
           "writing on":56}
NO_VERBS = 57   #verb数量

# 获取person和object的目标检测信息：box、score、class_no；图片的宽高shape
# 获取单个图片的标注信息[{'person_box': , 'hois': ['verb': ,'obj_box': ]}]
def get_detections_from_memory(segment_key, flag, detected_data, annotations):
    SCORE_TH = 0.6
    SCORE_OBJ = 0.3
    # 单个图片的标注信息
    annotation = annotations[str(segment_key)]
    # 单个图片文件名
    cur_obj_path_s = str(segment_key).replace('/','_')+'.json'

    # 单个图片的标注信息[{'person_box': , 'hois': ['verb': ,'obj_box': ]}]
    annotation = clean_up_annotation(annotation)
    # 单个图片的目标检测信息-------------------------需要修改。目前是在全部里根据关键字（图片文件名）获取
    detected_path = open(os.path.join(detected_data,cur_obj_path_s),'r')
    detections = json.load(detected_path)
    #detections = detected_data[cur_obj_path_s]

    img_H = detections['H']
    img_W = detections['W']
    shape = [img_W, img_H]
    # 过滤掉置信度低的目标，获得person和object的列表[box_coords,class_no,class_str,score]
    persons_d, objects_d = filter_bad_detections(detections, SCORE_TH, SCORE_OBJ)
    # 获取person和object的box、score、class_no列表
    d_p_boxes, scores_persons, class_id_humans = get_boxes_det(persons_d, img_H, img_W)
    d_o_boxes, scores_objects, class_id_objects = get_boxes_det(objects_d, img_H, img_W)
    scores_objects.insert(0, 1)
    return d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape


# 图片网格中每个小网格是否存在person和object返回列表[1, 2, H, W]
def get_attention_maps(tg):
    boxes = tg.get_field('boxes')   # np.concatenate((res['per_box'], res['obj_box'])) / [w, h, w, h]
    no_person_dets = tg.get_field('hoi_labels').shape[0]    #per number
    persons_np = boxes[:no_person_dets] # per box
    objects_np = boxes[no_person_dets:] # obj box
    labs = tg.get_field('mask')
    union_box = []
    pairs = tg.get_field('pairs_info')
    for cnt, pair in enumerate(pairs):
        dd_i = pair[0]  #person no
        do_i = pair[1] - no_person_dets # object no
        union_box.append(union_BOX(persons_np[dd_i], objects_np[do_i], obj_id=labs[cnt]))
    return torch.from_numpy(np.concatenate((union_box)))

'''根据图片id获取对应的目标检测和标注信息
    {'per_box': d_p_boxes_tensor, 
    "obj_box": d_o_boxes_tensor, 
    'labels_all': labels_np,    [dd=person_box数量,0,object为空的verb的id]=1:  [dd,object的索引+1=object数量,verb的id]=1
    'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)), 
    'labels_single': labels_single,     labels_np是否为空的标志
    'all_obj_labels': all_obj_classes, object_class_no
    'scores_persons': scores_persons, 
    'scores_objects': scores_objects,
    'shape': shape}'''
def get_anotation_info_by_imageId(segment_key, flag, detected_data, annotations):
    # 获取person和object的目标检测信息：box、score、class_no；图片的宽高shape
    # 获取单个图片的标注信息[{'person_box': , 'hois': ['verb': ,'obj_box': ]}]
    d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape = get_detections_from_memory(
        segment_key, flag, detected_data, annotations)
    if flag == 'test':
        MATCHING_IOU = 0.5
    else:
        MATCHING_IOU = 0.4
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    labels_np = np.zeros([no_person_dets, no_object_dets + 1, NO_VERBS], np.int32)  #person和object之间有动作变为1
    all_obj_classes = np.array([1] + class_id_objects)
    # 图片标注的person_box列表
    a_p_boxes = [ann['person_box'] for ann in annotation]
    # 计算person标注和目标检测的box之间的iou值列表-二维数组--要大于0.4
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)
    d_o_boxes_tensor = np.array([[0, 0, 0, 0]] + d_o_boxes, np.float32)
    '''if len(d_p_boxes)==0:
        d_p_boxes_tensor = np.array([[0, 0, 0, 0]], np.float32)
    else:'''
    d_p_boxes_tensor = np.array(d_p_boxes, np.float32)
    # 图片中存在person的box
    if no_person_dets != 0 and len(a_p_boxes) != 0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)  #iou最大值--[]
        index_for_each_det = np.argmax(iou_mtx, axis=0) #iou最大值索引
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_for_each_det[dd]  # 取iou值
            if cur_max_iou < MATCHING_IOU:  # 过滤
                continue
            matched_ann = annotation[index_for_each_det[dd]]    #获取对应的标注信息
            hoi_anns = matched_ann['hois']  #'hois': ['verb': ,'obj_box': ]
            # 未检测到object的标注列表['verb': ,'obj_box': ]
            noobject_hois = [oi for oi in hoi_anns if len(oi['obj_box']) == 0]

            # 未检测到object的verb的id。获得labels_np[dd,0,id]=1
            for no_hoi in noobject_hois:
                verb_idx = VERB2ID[no_hoi['verb']]
                labels_np[dd, 0, verb_idx] = 1

            # 能够检测到object的标注列表['verb': ,'obj_box': ]
            object_hois = [oi for oi in hoi_anns if len(oi['obj_box']) != 0]
            # object的box
            a_o_boxes = [oi['obj_box'] for oi in object_hois]
            # 计算object标注和目标检测的box之间的iou值列表-二维数组--要大于0.4
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)

            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                    for ao in range(len(a_o_boxes)):
                        cur_iou = iou_mtx_o[ao, do] # 取iou值
                        # enough iou
                        if cur_iou < MATCHING_IOU:
                            continue
                        current_hoi = object_hois[ao]   #object的标注列表['verb': ,'obj_box': ]
                        verb_idx = VERB2ID[current_hoi['verb']] #verb的索引
                        # 获得labels_np[dd,object的索引+1,verb的id]=1
                        labels_np[dd, do + 1, verb_idx] = 1  # +1 because 0 is no object

        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets + 1), NO_VERBS)
        # 如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true。labels_np是否为空的标志
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'per_box': d_p_boxes_tensor, "obj_box": d_o_boxes_tensor, 'labels_all': labels_np,
                'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)), 'labels_single': labels_single,
                'all_obj_labels': all_obj_classes, 'scores_persons': scores_persons, 'scores_objects': scores_objects,
                'shape': shape}
    else:
        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets + 1), NO_VERBS)
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'per_box': d_p_boxes_tensor, "obj_box": d_o_boxes_tensor, 'labels_all': labels_np,
                'all_obj_labels': all_obj_classes,
                'scores_persons': scores_persons, 'scores_objects': scores_objects, 'labels_single': labels_single,
                'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)), 'shape': shape}


# 图片网格中每个小网格是否存在person和object返回列表[1, 2, H, W]
def union_BOX(roi_pers, roi_objs, H=64, W=64, obj_id=0):
    assert H == W
    roi_pers = np.array(roi_pers * H, dtype=int)
    roi_objs = np.array(roi_objs * H, dtype=int)
    sample_box = np.zeros([1, 2, H, W])
    sample_box[0, 0, roi_pers[1]:roi_pers[3] + 1, roi_pers[0]:roi_pers[2] + 1] = 1
    sample_box[0, 1, roi_objs[1]:roi_objs[3] + 1, roi_objs[0]:roi_objs[2] + 1] = obj_id
    return sample_box

# 重置标注信息列表
# [{'person_box': , 'hois': ['verb': ,'obj_box': ]}]
def clean_up_annotation(annotation):
    persons_dict = {}
    for hoi in annotation:
        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
        dkey = tuple(box)
        objects = hoi['object']
        # 当前图片object为空
        if len(objects['obj_bbx']) == 0:  # no obj case
            cur_oi = {'verb': hoi['Verbs'],
                      'obj_box': [],
                      # 'obj_str': '',
                      }
        else:
            bbox=hoi['object']['obj_bbx'][0]
            cur_oi = {'verb': hoi['Verbs'],
                      'obj_box': [int(bbox[coord]) for coord in range(len(bbox))],
                      # 'obj_str': hoi['object']['obj_name'],
                      }

        # person的box坐标出现过，只改变verb、obj_box;
        # 否则添加person的box和object信息
        if dkey in persons_dict:
            persons_dict[dkey]['hois'].append(cur_oi)
        else:
            persons_dict[dkey] = {'person_box': box, 'hois': [cur_oi]}

    pers_list = []
    for dkey in persons_dict:
        pers_list.append(persons_dict[dkey])

    return pers_list

# 获取目标的box、score、class_no列表
def get_boxes_det(dets, img_H, img_W):
    boxes = []
    scores = []
    class_no = []
    for det in dets:
        top, left, bottom, right = det['box_coords']
        scores.append(det['score'])
        class_no.append(det['class_no'])
        left, top, right, bottom = left * img_W, top * img_H, right * img_W, bottom * img_H
        # left, top, right, bottom = left, top, right, bottom
        boxes.append([left, top, right, bottom])
    return boxes, scores, class_no

# 获取标注和目标检测给出的box之间的iou列表
def get_iou_mtx(anns, dets):
    no_gt = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_gt, no_dt])

    for gg in range(no_gt):
        gt_box = anns[gg]
        for dd in range(no_dt):
            dt_box = dets[dd]
            iou_mtx[gg, dd] = IoU_box(gt_box, dt_box)

    return iou_mtx

# 过滤掉置信度低的目标，获得person和object的列表[box_coords,class_no,class_str,score]
def filter_bad_detections(detections, SCORE_TH, SCORE_OBJ):
    persons = []
    objects = []
    for det in detections['detections']:
        if det['class_str'] == 'person':
            if det['score'] < SCORE_TH:
                continue
            persons.append(det)
        else:
            if det['score'] < SCORE_OBJ:
                continue
            objects.append(det)

    return persons, objects

# 计算两个box之间的iou
def IoU_box(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    left_int = max(left1, left2)
    top_int = max(top1, top2)
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU

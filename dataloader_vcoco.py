from __future__ import print_function, division
import json
import os
import torch
import pickle
import random
import numpy as np
from PIL import Image
import data_preprocess as dp
from skimage import io, transform
from utils import processing_sg, LIS
from transforms import build_transforms
from maskrcnn_benchmark.structures.bounding_box import BoxList
from torch.utils.data import Dataset
from maskrcnn_benchmark.structures.image_list import to_image_list

#Image list ids that cannot be detected any objects.
# 无法检测到任何对象的图像列表id。----------------需要获取
bad=open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/bad.json','r')
bad_data=json.load(bad)
bad_detections_train=bad_data['bad_detections_train']
bad_detections_test=bad_data['bad_detections_test']
bad_detections_val=bad_data['bad_detections_val']


def vcoco_collate(batch):
    transposed_batch = list(zip(*batch))
    images = to_image_list(transposed_batch[0], 32)
    targets = transposed_batch[1]
    return images, targets


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        transformed_img = transform.resize(image, (new_h, new_w))
        return transformed_img


class VcocoDataset(Dataset):
    def __init__(self, json_file_image, root_dir, cfg):
        with open(json_file_image) as json_file_:
            self.vcoco_frame_file = json.load(json_file_)
        # 判断数据集
        self.flag = json_file_image.split('/')[-1].split('_')[0]
        self.cfg = cfg
        # 获取检测图片、检测结果、实际标注信息
        if self.flag == 'train':
            # 能够检测到目标的图片编号列表
            print('111')
            self.vcoco_frame=[]
            for x in self.vcoco_frame_file.keys():
                if x not in str(bad_detections_train):
                    self.vcoco_frame.append(x)
            #self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_train)]
            # 目标检测结果-------------------只到Object_Detections 是否需要读取全体？
            print('222')
            self.detect_results = cfg.train_detected_results    # 改为路径
            #self.detect_results = json.load(open(cfg.train_detected_results, 'rb'))
            # 对象之间关系标注结果
            gt_path = cfg.data_dir + 'Annotations/train_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
            print('333')
        elif self.flag == 'val':
            self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_val)]
            self.detect_results = cfg.val_detected_results    # 改为路径
            #self.detect_results = pickle.load(open(cfg.train_detected_results, 'rb'))
            gt_path = cfg.data_dir + 'Annotations/test_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
        elif self.flag == 'test':
            self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_test)]
            self.detect_results = cfg.test_detected_results    # 改为路径
            #self.detect_results = pickle.load(open(cfg.test_detected_results, 'rb'))
            gt_path = cfg.data_dir + 'Annotations/test_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
        self.root_dir = root_dir    #图片目录
        # 读取预测的场景图--------整体路径
        self.sg_pred_path = os.path.join(cfg.sg_data , self.flag )
        # 图片尺寸变换 600×800
        self.transform = build_transforms(cfg, is_train=(self.flag == 'train'))
        self.max_nagetive = 512

    def __len__(self):
        return len(self.vcoco_frame)

    def convert2target(self, image, res):
        img_info = res['shape'] # 图片宽高
        w, h = img_info[0], img_info[1]
        #print(res['per_box'])
        #print(res['obj_box'])
        box = np.concatenate((res['per_box'], res['obj_box']))  # person、object的box
        scales = [w, h, w, h]
        box2 = box / scales
        box = torch.from_numpy(box).clone()
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        if self.transform is not None:
            image, target = self.transform(image, target)
        target.bbox[res['per_box'].shape[0], :] = 0
        per_obj_labs = np.concatenate((np.array([1 for _ in range(res['per_box'].shape[0])]), res['all_obj_labels']))
        target.add_field("labels", torch.from_numpy(per_obj_labs))
        target.add_field('boxes', torch.from_numpy(box2))
        per_scores = torch.tensor(res['scores_persons']).float()
        per_scores = LIS(per_scores, 8.3, 12, 10)

        obj_scores = torch.tensor(res['scores_objects']).float()
        obj_scores = LIS(obj_scores, 8.3, 12, 10)
        labels_scores = torch.cat((per_scores, obj_scores))

        target.add_field('obj_scores', labels_scores)
        pair_score = []

        pair_info = []
        hoi_labs = res['labels_all']
        target.add_field('hoi_labels', torch.from_numpy(hoi_labs))
        all_labels = target.get_field('labels')[res['labels_all'].shape[0]:]
        obj_label_for_mask = []
        one_hot_labs = []
        num_bg = 0
        for i in range(res['labels_all'].shape[0]): #person_box数量
            for j in range(res['labels_all'].shape[1]): #object_box数量
                if self.flag == 'test':
                    pair_info.append([i, res['labels_all'].shape[0] + j, 1])
                    one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                    pair_score.append(per_scores[i] * obj_scores[j])
                    obj_label_for_mask.append(all_labels[j])
                else:
                    if int(res['labels_all'][i, j, :].sum()) > 0:
                        pair_info.append([i, res['labels_all'].shape[0] + j, 1])
                        one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                        pair_score.append(per_scores[i] * obj_scores[j])
                        obj_label_for_mask.append(all_labels[j])
                    elif num_bg < self.max_nagetive:
                        pair_info.append([i, res['labels_all'].shape[0] + j, 0])
                        one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                        pair_score.append(per_scores[i] * obj_scores[j])
                        num_bg += 1
                        obj_label_for_mask.append(all_labels[j])
                    elif random.random() < 0.5:
                        replace_id = int(random.random() * len(pair_info))
                        pair_info[replace_id] = [i, res['labels_all'].shape[0] + j, 0]
                        one_hot_labs[replace_id] = target.get_field('hoi_labels')[i, j, :]
                        pair_score[replace_id] = per_scores[i] * obj_scores[j]
                        obj_label_for_mask[replace_id] = all_labels[j]

        target.add_field("pairs_info", torch.tensor(pair_info))
        target.add_field("mask", torch.tensor(obj_label_for_mask))
        target.add_field("per_mul_obj_scores", torch.tensor(pair_score))
        target.add_field('HOI_labs', torch.stack(one_hot_labs))

        return image, target

    def __getitem__(self, idx):
        img_pre_suffix = str(self.vcoco_frame[idx]).replace('/','_') + '.json'
        img_path=str(self.vcoco_frame[idx]).replace('png','jpg')
        img_name = os.path.join(self.root_dir, img_path)  #单张图片路径
        image = Image.open(img_name).convert('RGB')
        '''根据图片id获取对应的目标检测和标注信息
        {'per_box': d_p_boxes_tensor,
         "obj_box": d_o_boxes_tensor,
         'labels_all': labels_np, [dd, 0, object为空的verb的id] = 1:  [dd, object的索引 + 1, verb的id] = 1
        'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)),
        'labels_single': labels_single, labels_np是否为空的标志
        'all_obj_labels': all_obj_classes,
        'scores_persons': scores_persons,
        'scores_objects': scores_objects,
        'shape': shape}'''
        all_info = dp.get_anotation_info_by_imageId(self.vcoco_frame[idx], self.flag, self.detect_results,self.annotations)
        # 获取图片image经过transfrom后结果；图片对应的目标检测和标注信息
        image, target = self.convert2target(image, all_info)
        # 图片网格中每个小网格是否存在person和object返回列表[1, 2, H, W]
        union_box = dp.get_attention_maps(target)
        target.add_field('union_box', torch.tensor(union_box).float())
        mp4=self.vcoco_frame[idx].split('.mp4/')[0]
        self.sg_pred=json.load(open(os.path.join(self.sg_pred_path,mp4+'_SGG.json'),'r'))
        target.add_field('sg', self.sg_pred[self.vcoco_frame[idx]])    # 场景图[图片名]--该图片对应的场景图
        # 获取处理后的场景图{'relations': valid_rels, 'entities': entities}
        sg_graph = processing_sg(self.sg_pred[self.vcoco_frame[idx]], target)
        target.add_field('sg_data', sg_graph)
        target.add_field('image_id', self.vcoco_frame[idx])

        return image, target

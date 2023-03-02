from __future__ import print_function, division
import torch
import argparse
import os
import torch.optim as optim
import warnings
import time
import pickle

warnings.filterwarnings("ignore")

import sys
#sys.path.append('/home/qiujin/maskrcnn/maskrcnn-benchmark')
from model import SG2HOI
import datetime
from apex import amp
import pandas as pd
from tqdm import tqdm
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from util.metric_logger import MetricLogger
from util.checkpoint import clip_grad_norm
from solver.trainer import reduce_loss_dict
from torch.utils.data import DataLoader
from dataloader_vcoco import VcocoDataset, vcoco_collate
from utils import mkdir, debug_print, get_side_infos, setup_logger, save_checkpoint, \
    calculate_averate_precision
import sys
# 创建模型、加载参数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_epochs', type=int, required=False, default=100)
    parser.add_argument('--gpu_id', type=str, required=False, default="0", )
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01, help='Initial learning_rate')
    parser.add_argument('--saving_epoch', '--saving_epoch', type=int, required=False, default=10)
    # 输出路径
    parser.add_argument('--output', type=str, required=False, default='/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Output')
    parser.add_argument('--batch_size', type=int, required=False, default=5, help='Batch size')
    parser.add_argument('--resume_model', type=bool, required=False, default=False) # 改为False
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-num_work', '--num_work', type=int, default=5, required=False,
                        help="number of threads for data_loader.")
    parser.add_argument('-mean_best', '--mean_best', type=float, default=0., required=False)
    parser.add_argument('-start_epoch', '--start_epoch', type=int, default=0, required=False)
    parser.add_argument('-num_epochs', '--num_epochs', type=int, default=50, required=False)
    parser.add_argument('-device', '--device', type=str, default="cuda", required=False)
    parser.add_argument('--data_dir', type=str, default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/", required=False)
    parser.add_argument('--prior', type=str, default="/mnt/hdd2/hoi_data/infos/prior.pickle", required=False)
    # 目标检测结果
    parser.add_argument('--train_detected_results', type=str,
                        default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/train/",
                        required=False)
    parser.add_argument('--val_detected_results', type=str,
                        default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/test/",
                        required=False)
    parser.add_argument('--test_detected_results', type=str,
                        default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/test/",
                        required=False)
    # 场景图预测路径
    parser.add_argument('--sg_data', type=str,
                        default="/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/SGG/", required=False,help="Your scene graph predicted results' path.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    # 加载数据
    data_loaders = construct_dataloaders(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda")
    # 创建模型
    model = SG2HOI(args).to(device)
    logger = setup_logger("SG2HOI", args.output, 0)  # 日志
    trainables = []
    for name, p in model.named_parameters():
        if name.split('.')[0] in ['Conv_pretrain']:
            p.requires_grad = False
        else:
            trainables.append(p)
    if args.output:
        mkdir(args.output)
    # 随机梯度下降
    optimizer = optim.SGD([{"params": trainables, "lr": args.learning_rate}], momentum=0.9, weight_decay=0.0001)
    lambd = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 33 else 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lambd])
    debug_print(logger, 'end optimizer and shcedule')
    amp_opt_level = 'O0'  # "O1" for mixture precision training.
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    # 打印训练过程中信息
    meters = MetricLogger(delimiter="  ")

    if model.use_faster_rcnn_backbone:
        # 未使用faster_rcnn作为特征提取的主网络
        load_mapping = {"box_feature_extractor": "roi_heads.box.feature_extractor", 'backbone_net': 'backbone'}
        checkpoint = torch.load('/mnt/hdd2/datasets/vg/pretrained_faster_rcnn/model_final.pth',
                                map_location=torch.device("cpu"))
        load_state_dict(model, checkpoint.pop("model"), load_mapping)

    if args.resume_model is True:
        # 进入该分支
        try:
            # 已保存的最好的模型结果
            checkpoint = torch.load(args.output + '/bestcheckpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args.start_epoch = checkpoint['epoch']
            args.mean_best = checkpoint['mean_best']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(
                "=> loaded checkpoint when best_prediction {} and epoch {}".format(args.mean_best, checkpoint['epoch']))
        except:
            print('Failed to load checkPoint')
            raise

    train_test(model, optimizer, scheduler, data_loaders, logger, meters, args)

# 加载数据
def construct_dataloaders(cfg):
    # 人和物品及其动作标注信息
    annotation_train = cfg.data_dir + 'Annotations/train_annotations.json'
    # 图片
    image_dir_train = cfg.data_dir + 'Image/train/'

    annotation_val = cfg.data_dir + 'Annotations/test_annotations.json'
    image_dir_val = cfg.data_dir + 'Image/test/'

    annotation_test = cfg.data_dir + 'Annotations/test_annotations.json'
    image_dir_test = cfg.data_dir + 'Image/test/'

    # 读取图片列表、目标检测、实际标注、场景图
    print('VcocoDataset start')
    vcoco_train = VcocoDataset(annotation_train, image_dir_train, cfg)
    vcoco_val = VcocoDataset(annotation_val, image_dir_val, cfg)
    vcoco_test = VcocoDataset(annotation_test, image_dir_test, cfg)
    num_workers = cfg.num_work
    dataloader_train = DataLoader(vcoco_train, cfg.batch_size, shuffle=True, collate_fn=vcoco_collate,
                                  num_workers=num_workers, )
    dataloader_val = DataLoader(vcoco_val, cfg.batch_size, shuffle=True, collate_fn=vcoco_collate,
                                num_workers=num_workers, )
    dataloader_test = DataLoader(vcoco_test, 1, shuffle=False, collate_fn=vcoco_collate, num_workers=num_workers, )
    dataloader = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
    return dataloader

# 执行测试
def run_test(model, data_loader, device, split='test'):
    model.eval()
    true_labels = []
    predicted_scores = []

    person_boxes = []
    object_boxes = []
    img_ids = []
    nums_o_p_infor = []
    class_ids = []
    for image, targets in tqdm(data_loader[split]):
        image = image.to(device)
        # 获取实际信息
        person_boxes_, object_boxes_, class_ids_, img_ids_, nums_o_p_info_ = get_side_infos(targets)
        person_boxes.append(person_boxes_)
        object_boxes.append(object_boxes_)
        class_ids.append(class_ids_)

        img_ids += img_ids_
        nums_o_p_infor += nums_o_p_info_

        targets = [t.to(device) for t in targets]
        # 预测
        predicted_hoi, _ = model(image, targets)
        predicted_scores.append(predicted_hoi.data.cpu())
        labels = torch.cat([t.get_field('HOI_labs') for t in targets]).data.cpu()
        true_labels.append(labels)

    predicted_scores = torch.cat(predicted_scores).numpy()
    true_labels = torch.cat(true_labels).numpy()
    # 评估--根据预测分数计算平均净度AP
    ap_results = calculate_averate_precision(predicted_scores, true_labels)
    detections_test = []

    ap_test = pd.DataFrame(ap_results, columns=['TEST', 'Score_TEST'])
    model.train()
    return ap_test, detections_test

# 开始训练
def train_test(model, optimizer, scheduler, dataloader, logger, meters, cfg):
    torch.cuda.empty_cache()
    training_phases = ['train', 'val']
    iteration = 0
    mean_best = cfg.mean_best
    end = time.time()
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        scheduler.step()
        for phase in training_phases:
            model.train()
            true_scores = []
            predicted_scores = []
            torch.cuda.empty_cache()
            # 训练、测试数据加载器
            for image, targets in dataloader[phase]:
                data_time = time.time() - end
                image = image.to(cfg.device)
                # 实际标注
                targets = [t.to(cfg.device) for t in targets]
                # 预测
                predicted_hoi, loss_dict = model(image, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)
                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)
                eta_seconds = meters.time.global_avg * (100000 - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                optimizer.zero_grad()
                with amp.scale_loss(losses, optimizer) as scaled_losses:
                    scaled_losses.backward()
                clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                               max_norm=5, logger=logger, verbose=(iteration % 500 == 0), clip=True)
                optimizer.step()
                # 写入日志
                if iteration % 50 == 0:
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}"

                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer.param_groups[-1]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

                predicted_scores.append(predicted_hoi)
                true_scores.append(torch.cat([t.get_field('HOI_labs') for t in targets]))
                iteration += 1
            # 预测得分、实际得分
            predicted_scores = torch.cat(predicted_scores).data.cpu().numpy()
            true_scores = torch.cat(true_scores).data.cpu().numpy()

            # 评估
            if phase == 'train':
                # 根据预测分数计算平均精度AP，将其展示为表格形式并输出
                ap_train = calculate_averate_precision(predicted_scores, true_scores)
                ap_train = pd.DataFrame(ap_train, columns=['Name_TRAIN', 'Score_TRAIN'])
                print(ap_train)

            elif phase == 'val':
                ap_test = calculate_averate_precision(predicted_scores, true_scores)
                ap_test = pd.DataFrame(ap_test, columns=['Name_VAL', 'Score_VAL'])
                print(ap_test)

        # 每3个epoch进行一个测试
        if epoch % 3 == 0 and epoch >= 0:
            resutls, detections_test = run_test(model, dataloader, cfg.device)
            print(resutls)

            # ------------目标检测结果 但是为[]？
            file_name_p = cfg.output + '/' + 'test{}.pickle'.format(epoch + 1)
            with open(file_name_p, 'wb') as handle:
                pickle.dump(detections_test, handle)

            mean_resutls = resutls.to_records(index=False)[29][1]
            # 测试结果更好，保存模型信息
            if mean_resutls > mean_best:
                mean_best = mean_resutls
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'mean_best': mean_best,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, filename=cfg.output + '/' + 'bestcheckpoint.pth.tar')

        # 每10个epoch保存一次模型信息
        if (epoch + 1) % cfg.saving_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_best': mean_best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, filename=cfg.output + '/model_' + str(epoch + 1).zfill(6) + '_checkpoint.pth.tar')


if __name__ == "__main__":
    main()

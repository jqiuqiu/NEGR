import array
import os
import zipfile
import itertools
import six
import json
import torch
import torch.nn as nn
import numpy as np
import logging
from six.moves.urllib.request import urlretrieve
from tqdm import tqdm
import sys
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from maskrcnn_benchmark.config import cfg


VERBS_NO_COCO = 80

VERB2ID = [#verb
    "awakening","closing","cooking","dressing","drinking","fixing","grasping","laughing","lying","making",
    "opening","photographing","playing on","pouring","putting down","running","sitting","smiling","sneezing",
    "snuggling","standing","taking","talking","throwing","tidying","turning","undressing","walking",
    "washing","watching","working on",
	#relatioin
	"looking at","not looking at","unsure","above","beneath","in front of","behind","on the side of","in",
	"carrying","covered by","drinking from","eating","having it on the back","holding","leaning on","lying on",
	"not contacting","other relationship","sitting on","standing on","touching","twisting","wearing","wiping",
    "writing on"]


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."
    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups
    return group_gn

def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )

def make_fc(dim_in, hidden_dim, use_gn=False):

    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(fc.weight, a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


# 根据预测分数计算平均精度
def calculate_averate_precision(predicted_score, true_score):
    result = []
    mean = 0
    for k in range(len(VERB2ID)):
        predicted = predicted_score[:, k]
        true = true_score[:, k]
        try:
            # 根据预测分数计算平均精度
            class_ap = average_precision_score(true, predicted) * 100
        except:
            print('evaluation error!')
            raise
        mean += class_ap
        result.append((VERB2ID[k], class_ap))
    result.append(('All-Mean', mean / len(VERB2ID)))

    return result

# 获取两个box的并集和交集区域
def get_box_pair_info(box1, box2):
    """
    input:
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output:
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    unionbox = box1[:, :4].clone()
    unionbox[:, 0] = torch.min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch.min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch.max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch.max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)

    # intersection box
    intersextion_box = box1[:, :4].clone()
    intersextion_box[:, 0] = torch.max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch.max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch.min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch.min(box1[:, 3], box2[:, 3])
    case1 = torch.nonzero(
        intersextion_box[:, 2].contiguous().view(-1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch.nonzero(
        intersextion_box[:, 3].contiguous().view(-1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch.cat((box1, box2, union_info, intersextion_info), 1)

# 提取场景图中每个对象的空间特征pi
def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


def normalize_sigmoid_logits(orig_logits):
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits


def transpose_packed_sequence_inds(lengths):
    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(path, ' is existed.')


def debug_print(logger, info):
    logger.info('#' * 20 + ' ' + info + ' ' + '#' * 20)


def center_x(proposals):
    assert proposals[0].mode == 'xyxy'
    boxes = torch.cat([p.bbox for p in proposals], dim=0)
    c_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    return c_x.view(-1)


def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1, 1], dim=-1)
        x, y = xy.split([1, 1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1, 1, 1, 1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w / wid, h / hei, x / wid, y / hei, x1 / wid, y1 / hei, x2 / wid, y2 / hei,
                          w * h / (wid * hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)

#relation词向量
def rel_word_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)
    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        elif token != 'walking on':
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))
        else:
            wv_index = wv_dict.get('roam', None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
                print("{} -> {} ".format(token, 'roam'))
            else:
                print("fail on {}".format(token))

    return vectors

# 返回names列表中每个词的词向量
def obj_edge_vectors(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    # 词及对应索引的字典，词向量数组，词向量长度300
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    #遍历需要获取词向量的数组列表
    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None) # 词向量所在索引
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]   # 获取词向量
        elif token != 'walking on': # 词不在数组中，并且这个词不是lw_token
            # Try the longest word 按空格划分后进行排序得到lw_token
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)  # 在数组中找lw_token
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))   #还没找到就是失败
        else:   #如果词是walking on 将它替换成roam，找对应的向量
            wv_index = wv_dict.get('roam', None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
                print("{} -> {} ".format(token, 'roam'))
            else:
                print("fail on {}".format(token))
    return vectors

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

# 加载词向量
def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim) # fname=glove.6B.300d

    # 非第一次执行，进入此分支
    if os.path.isfile(fname + '.pt'):
        # 加载词向量从glove.6B.300d.pt
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    if os.path.isfile(fname + '.txt'):
        # 读取glove.6B.300d.txt文件
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]  # cm是每个词对应的glove向量
    # 第一次执行进入此分支
    elif os.path.basename(wv_type) in URL:
        # 根据URL路径下载词向量，并再次调用函数读取该词向量
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        # 下载词向量
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None # array.array('d')创建一个float型的数组
    if cm is not None:
        # cm长度是400000
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')  # 空格分割
            word, entries = entries[0], entries[1:] # 词word是第一个，向量entries是之后的所有
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)    #将所有的向量添加到词向量数组
            wv_tokens.append(word)  #词的数组

    wv_dict = {word: i for i, word in enumerate(wv_tokens)} # 词及对应的索引组成的字典
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size) # 词向量数组
    ret = (wv_dict, wv_arr, wv_size)    # 词及对应的索引，词向量，词向量长度300
    torch.save(ret, fname + '.pt')  # 将获取到的词向量保存到glove.6B.300d.pt
    return ret


def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch.cat((boxes, center_box), 1)
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return


def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch.min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch.max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inters = inter[:, :, :, 0] * inter[:, :, :, 1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
            boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inters + areas[None] + areas[:, None]
    return inters / union


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0),
                                                  boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh

    prob_sampled = F.softmax(pred_logits, 1).cpu().numpy()
    prob_sampled[:, 0] = 0  # set bg to 0

    pred_label = torch.zeros(num_obj, device=pred_logits.device, dtype=torch.int64)

    for i in range(num_obj):
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0  # This way we won't re-sample

    return pred_label


def LIS(x, T, k, w):
    return T / (1 + torch.exp(k - w * x))

# 场景图处理---------------------需要修改
# {'relations': valid_rels, 'entities': entities}
def processing_sg(sg_mat, target):
    # 某图片的场景图，目标
    valid_rels = []
    LAB_SET = target.get_field('labels')
    entities = [0]
    # pairs = target.get_field('pairs_info')
    #print(pairs)

    relation=load_all_vocabs()
    rel_attention_labels=sg_mat['rel_attention_labels']
    for i in range(len(rel_attention_labels)):
        rel_attention_label=rel_attention_labels[i]
        if len(rel_attention_label) != 0:
            for j in range(len(rel_attention_label)):
                #--------------------------是否需要-1
                number=torch.from_numpy(np.array(relation.index(rel_attention_label[j])+1))
                valid_rels.append([0, LAB_SET[i+1] - 1, number])
                #valid_rels.append([0, LAB_SET[i+1], rel_contact_label[j]])
        if LAB_SET[i+1] - 1 not in entities:
            entities.append(LAB_SET[i+1] - 1)

    rel_contact_labels=sg_mat['rel_contact_labels']
    for i in range(len(rel_contact_labels)):
        rel_contact_label=rel_contact_labels[i]
        if len(rel_contact_label) != 0:
            for j in range(len(rel_contact_label)):
                #--------------------------是否需要-1
                number=torch.from_numpy(np.array(relation.index(rel_contact_label[j])+1))
                valid_rels.append([0, LAB_SET[i+1] - 1, number])
                #valid_rels.append([0, LAB_SET[i+1], rel_contact_label[j]])
        if LAB_SET[i+1] - 1 not in entities:
            entities.append(LAB_SET[i+1] - 1)

    rel_spatial_labels=sg_mat['rel_spatial_labels']
    for i in range(len(rel_spatial_labels)):
        rel_spatial_label=rel_spatial_labels[i]
        if len(rel_spatial_label) != 0:
            for j in range(len(rel_spatial_label)):
                #--------------------------是否需要-1
                number=torch.from_numpy(np.array(relation.index(rel_spatial_label[j])+1))
                valid_rels.append([0, LAB_SET[i+1] - 1, number])
                #valid_rels.append([0, LAB_SET[i+1], rel_contact_label[j]])
        if LAB_SET[i+1] - 1 not in entities:
            entities.append(LAB_SET[i+1] - 1)

    rel_verb_labels=sg_mat['rel_verb_labels']
    for i in range(len(rel_verb_labels)):
        rel_verb_label=rel_verb_labels[i]
        if len(rel_verb_label) != 0:
            for j in range(len(rel_verb_label)):
                #--------------------------是否需要-1
                number=torch.from_numpy(np.array(relation.index(rel_verb_label[j])+1))
                valid_rels.append([0, LAB_SET[i+1] - 1, number])
                #valid_rels.append([0, LAB_SET[i+1], rel_contact_label[j]])
        if LAB_SET[i+1] - 1 not in entities:
            entities.append(LAB_SET[i+1] - 1)
    entities = torch.from_numpy(np.array(entities))
    valid_rels = torch.from_numpy(np.array(valid_rels))
    sg_graph = {'relations': valid_rels, 'entities': entities}
    return sg_graph


def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        torch.nn.init.orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]


def construct_pair_feature(pers, objs, sg_ctx, targets):
    pairs_out = []
    sg_feats = []
    pair_bboxs_info = []

    start = 0
    start_p = 0
    start_o = 0
    for batch in range(len(targets)):

        target = targets[batch]
        # 获取person object特征信息
        num_pers = target.get_field('hoi_labels').shape[0]
        num_objs = target.get_field('hoi_labels').shape[1]
        num_all = num_pers + num_objs
        batch_pers, batch_objs = pers[start_p:start_p + num_pers], objs[start_o:start_o + num_objs]
        pers_objs = []
        sg_single = target.get_field('sg')  # 场景图
        sg_adj_per2obj = np.zeros((len(batch_pers), len(batch_objs) - 1, 57), dtype=np.float32)
        sg_adj_obj2per = np.zeros((len(batch_objs) - 1, len(batch_pers), 57), dtype=np.float32)
        pair_idx = target.get_field('pairs_info')   # 关系对
        # 提取场景图中每个对象的空间特征pi [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        obj_boxes = get_box_info(target.bbox, need_norm=True, proposal=target)

        # 获取关系对中两个box的并集和交集区域 [box1, box2, unionbox, intersectionbox]
        pair_bboxs_info.append(get_box_pair_info(obj_boxes[pair_idx[:, 0].long()], obj_boxes[pair_idx[:, 1].long()]))
        # 用glove提取语言特征


        # 获取邻域矩阵
        for pair in pair_idx:
            # 连接一对关系中person和object的特征
            pair = pair.long()
            hum_feat = batch_pers[pair[0]]
            obj_feat = batch_objs[pair[1] - num_pers]
            pers_objs.append(torch.cat([hum_feat, obj_feat], 0))

            if pair[1] - num_pers > 0:
                ind_p = pair[0]
                ind_o = pair[1]
                #得到person->object和object->person的场景图矩阵---------------------------需要改
                sg_vc = sg_single[ind_p][ind_o]
                sg_adj_per2obj[ind_p, ind_o - 1 - num_pers, :] = sg_vc
                sg_vc2 = sg_single[ind_o][ind_p]
                sg_adj_obj2per[ind_o - 1 - num_pers, ind_p, :] = sg_vc2

        sg_feats.append([torch.from_numpy(sg_adj_per2obj).to(batch_pers.device),
                         torch.from_numpy(sg_adj_obj2per).to(batch_pers.device)])

        # 关系中person和object的特征建立堆栈
        pers_objs_batch = torch.stack(pers_objs)
        # 一张场景图得到关系信息
        if sg_ctx is not None:# 进入
            batch_context = sg_ctx[batch]   # 具有上下文的场景图信息
            pairs_out.append(torch.cat([pers_objs_batch, batch_context.repeat(pers_objs_batch.size()[0], 1)], 1))
        else:
            pairs_out.append(pers_objs_batch)
        start += num_all
        start_p += num_pers
        start_o += num_objs
    # 具有上下文信息得到全局场景图表示以及用于最终预测的人和物的初始特征、场景图的邻域矩阵（用于消息传递）、提取场景图中每个对象的空间特征pi以及获取两个box的并集和交集区域
    return torch.cat(pairs_out), sg_feats, torch.cat(pair_bboxs_info)

# 获取所有的verb
def load_all_vocabs():
    vocab_file = json.load(open('/home/qiujin/LiVLR/LiVLR-VideoQA-main/data/AGQA/AG/Object_Detections/VG-SGG-dicts.json'))
    predicates = vocab_file['idx_to_predicate']
    preds = []
    for i in range(1, 58):
        preds.append(predicates[str(i)])
    return preds

# 将对象划分为person object
def split_to_hum_obj(rois, tgts):
    people_rois = []
    obj_rois = []
    offset = 0
    for batch in range(len(tgts)):
        (N_hum, N_obj, _) = tgts[batch].get_field('hoi_labels').shape
        img_rois = rois[offset:offset + N_hum + N_obj]
        people_rois.append(img_rois[:N_hum])
        obj_rois.append(img_rois[N_hum:])
        offset += N_hum + N_obj
    return torch.cat(people_rois), torch.cat(obj_rois)


def convert_to_roi_format(boxes):
    concat_boxes = torch.cat([b.bbox for b in boxes], dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [
            torch.full((len(b), 1), i, dtype=dtype, device=device)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)

    return rois

# 设置日志信息
def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def convert2vcoco(predicted_HOI, persons_np, objects_np, pairs_info, image_id, class_ids):
    vcoco_map = {}
    num_img = 0
    increment = [int(i[0] * i[1]) for i in pairs_info]
    start = 0
    scores = []
    p_boxes = []
    o_boxes = []
    cls_ids = []
    for index in tqdm(range(predicted_HOI.shape[0])):
        scores.append(predicted_HOI[index])
        p_boxes.append(persons_np[index])
        o_boxes.append(objects_np[index])
        cls_ids.append(class_ids[index])
        if index == start + increment[num_img] - 1:
            vcoco_map[image_id[num_img], 'score'] = np.stack(scores)
            vcoco_map[image_id[num_img], 'pers_bbx'] = np.stack(p_boxes)
            vcoco_map[image_id[num_img], 'obj_bbx'] = np.stack(o_boxes)
            vcoco_map[image_id[num_img], 'class_ids'] = np.array(cls_ids).reshape(-1, 1)
            start += increment[num_img]
            num_img += 1
            scores = []
            p_boxes = []
            o_boxes = []
            cls_ids = []
    return vcoco_map


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# 获取human_box、object_box、类别ID、图片ID、人与对象的交互标签
def get_side_infos(targets):
    person_boxes = []
    object_boxes = []
    img_ids = []
    nums_o_p_infor = []
    class_ids = []
    for tg in targets:
        class_ids.append(tg.get_field('mask').data.cpu().numpy())
        pairs = tg.get_field('pairs_info').data.cpu().numpy()
        boxes = tg.get_field('boxes').data.cpu().numpy()

        person_boxes.append(boxes[pairs[:, 0]])
        object_boxes.append(boxes[pairs[:, 1]])
        img_ids.append(int(tg.get_field('image_id')))
        hum_n, obj_n = tg.get_field('hoi_labels').shape[0], tg.get_field('hoi_labels').shape[1]
        nums_o_p_infor.append([hum_n, obj_n])
    return np.concatenate(person_boxes), np.concatenate(object_boxes), np.concatenate(
        class_ids), img_ids, nums_o_p_infor


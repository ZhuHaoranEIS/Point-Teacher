import json
import os

import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import random

import mmcv
from aitod_dataset import AitodDataset
from xview_dataset import XviewDataset
from dior_dataset import DiorDataset
from dotav2_dataset import Dotav2Dataset

from pycocotools.coco import COCO

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径

def merge_coco_files(folder_path):
    merged_data = {
        "info": {
            "year": 2024,
            "version": "1",
            "date_created": "1.27"
        },
        "images": [],
        "annotations": [],
        "licenses": [
            {
                "id": 1,
                "name": "zhuhaoran",
                "url": "null"
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "vehicle",
                "supercategory": "none"
            }
        ]
    }

    image_id_counter = 0
    xview_category = ['Bus', 'Cargo Truck', 'Cargo/Container Car', 'Cement Mixer', 'Crane Truck', 'Dump Truck', 'Engineering Vehicle', 'Excavator', 'Flat Car', 'Front loader/Bulldozer',
     'Haul Truck', 'Passenger Vehicle', 'Passenger Car', 'Pickup Truck', 'Railway vehicle', 'Small Car', 'Straddle Carrier', 'Tank car', 'Tower crane', 'Tractor', 'Trailer',
     'Truck', 'Truck Tractor', 'Truck Tractor w/ Box Trailer', 'Truck Tractor w/ Flatbed Trailer', 'Truck Tractor w/ Liquid Tank', 'Utility Truck']
    xview_id = [4, 5, 6, 7, 11, 13, 14, 15, 20, 21, 23, 32, 33, 35, 37, 44, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 58]
    json_all = ['/data/zhr/DIOR/trainval.json', '/data/zhr/DIOR/test.json', 
                '/data/zhr/AI-TOD/AI-TOD-v2/aitodv2_train.json', '/data/zhr/AI-TOD/AI-TOD-v2/aitodv2_val.json',
                '/data/zhr/DOTAv2/hbb/annotations/dota_v2.0_train.json', '/data/zhr/DOTAv2/hbb/annotations/dota_v2.0_val.json']
    images_all = ['/data/xc/DIOR/JPEGImages-trainval', '/data/xc/DIOR/JPEGImages-test',
                  '/data/zhr/AI-TOD/AI-TOD-v1/train', '/data/zhr/AI-TOD/AI-TOD-v1/val',
                  '/data/zhr/DOTAv2/hbb/train', '/data/zhr/DOTAv2/hbb/val']
    format_all = ['jpg', 'jpg',
                  'png', 'png',
                  'png', 'png']
    classname_all = [['vehicle'], ['vehicle'],
                     ['vehicle'], ['vehicle'],
                     ['small-vehicle', 'large-vehicle'], ['small-vehicle', 'large-vehicle']]
    
    id = [[2], [2],
          [5], [5],
          [5, 6], [5, 6]]

    save_temp = '/home/zhr/mmdet-rfla/tools/dataset_converters/'
    
    dior = DiorDataset()
    aitod = AitodDataset()
    dotav2 = Dotav2Dataset()
    xview = XviewDataset()
    dataset_all = [dior, dior,
                   aitod, aitod,
                   dotav2, dotav2,
                   xview, xview]
    img_save_path = '/data/zhr/spc_vehicle_small/images/'

    max_width = 0
    min_width = 0
    max_height = 0
    min_height = 0
    img_name_all = []
    
    new_data_infos_train = []
    new_data_infos_val = []
    list_train_val = [0, 1]
    prob_train_val = [0.8, 0.2]
    for i in tqdm(range(len(json_all))):
        json_path = json_all[i]
        image_path = images_all[i]
        image_format = format_all[i]
        anno_temp = dataset_all[i].load_annotations(json_path)
        class_name_temp = dataset_all[i].CLASSES
        id_temp = id[i]
        save_path_temp = save_temp + 'temp.json'
        mmcv.dump(anno_temp, save_path_temp)
        data_infos = mmcv.load(save_path_temp)
        # print('\ngenerating new ground-truth ...' + str(i))
        data_len = len(data_infos)
        for idx in tqdm(range(data_len)):
            train_or_val = random.choices(list_train_val, prob_train_val)
            # import pdb; pdb.set_trace()
            img_w, img_h = data_infos[idx]['width'], data_infos[idx]['height']
            if i == 0 and idx == 0:
                max_width = img_w
                max_height = img_h
                min_width = img_w
                min_height = img_h
            else:
                if img_w > max_width:
                    max_width = img_w
                if img_h > max_height:
                    max_height = img_h
                if img_w < min_width:
                    min_width = img_w
                if img_h < min_height:
                    min_height = img_h

            anno = data_infos[idx]['ann']
            filename = data_infos[idx]['filename']

            bboxes = anno['bboxes']
            category_id = anno['labels']
            new_bboxes = []
            new_category = []
            new_category_name = []
            ori_category_name = []
            new_iscrowd = []
            new_ignore = []
            save_or_not = False
            for j, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2
                cate = category_id[j]
                cate_name = class_name_temp[cate]
                if cate in id_temp:
                    save_or_not = True
                    new_bboxes.append([x, y, w, h])
                    new_category.append(0)
                    new_category_name.append('vehicle')
                    ori_category_name.append(cate_name)
                    if 'iscrowd' in anno:
                        new_iscrowd.append(anno['iscrowd'][j])
                    else:
                        new_iscrowd.append(0)
                    if 'ignore' in anno:
                        new_ignore.append(anno['ignore'][j])
                    else:
                        new_ignore.append(0)
            if (save_or_not == True) and (filename not in img_name_all):
                img_name_all.append(filename)
                data_infos_temp = {}
                data_infos_temp['id'] = image_id_counter
                image_id_counter = image_id_counter + 1
                data_infos_temp['file_name'] = filename
                data_infos_temp['width'] = img_w
                data_infos_temp['height'] = img_h
                data_infos_temp['license'] = "None"
                data_infos_temp['flickr_url'] = "None"
                data_infos_temp['coco_url'] = "None"
                data_infos_temp['filename'] = filename
                data_infos_temp['ann'] = {}

                data_infos_temp['ann']['bboxes'] = np.array(new_bboxes).astype(np.float32)
                data_infos_temp['ann']['labels'] = np.array(new_category).astype(np.int32)
                data_infos_temp['ann']['new_category_name'] = np.array(new_category_name)
                data_infos_temp['ann']['ori_category_name'] = np.array(ori_category_name)
                data_infos_temp['ann']['iscrowd'] = np.array(new_iscrowd).astype(np.int32)
                data_infos_temp['ann']['ignore'] = np.array(new_ignore).astype(np.int32)
                if train_or_val[0] == 0:
                    new_data_infos_train.append(data_infos_temp)
                    ori_image_path = os.path.join(image_path, filename)
                    new_image_path = os.path.join('/data/zhr/spc_vehicle_small/image/train', filename)
                else:
                    new_data_infos_val.append(data_infos_temp)
                    ori_image_path = os.path.join(image_path, filename)
                    new_image_path = os.path.join('/data/zhr/spc_vehicle_small/image/val', filename)
                # save image
                # ori_image_path = os.path.join(image_path, filename)
                # new_image_path = os.path.join(img_save_path, filename)
                shutil.copy(ori_image_path, new_image_path)
            elif (save_or_not == True) and (filename in img_name_all):
                new_filename = filename.split('.')[0] + '_' + str(i) + '_' + str(idx) + '_' + str(j) + '.' + filename.split('.')[-1]
                img_name_all.append(new_filename)
                data_infos_temp = {}
                data_infos_temp['id'] = image_id_counter
                image_id_counter = image_id_counter + 1
                data_infos_temp['file_name'] = new_filename
                data_infos_temp['width'] = img_w
                data_infos_temp['height'] = img_h
                data_infos_temp['license'] = "None"
                data_infos_temp['flickr_url'] = "None"
                data_infos_temp['coco_url'] = "None"
                data_infos_temp['filename'] = new_filename
                data_infos_temp['ann'] = {}

                data_infos_temp['ann']['bboxes'] = np.array(new_bboxes).astype(np.float32)
                data_infos_temp['ann']['labels'] = np.array(new_category).astype(np.int32)
                data_infos_temp['ann']['new_category_name'] = np.array(new_category_name)
                data_infos_temp['ann']['ori_category_name'] = np.array(ori_category_name)
                data_infos_temp['ann']['iscrowd'] = np.array(new_iscrowd).astype(np.int32)
                data_infos_temp['ann']['ignore'] = np.array(new_ignore).astype(np.int32)

                if train_or_val[0] == 0:
                    new_data_infos_train.append(data_infos_temp)
                    ori_image_path = os.path.join(image_path, filename)
                    new_image_path = os.path.join('/data/zhr/spc_vehicle_small/image/train', new_filename)
                else:
                    new_data_infos_val.append(data_infos_temp)
                    ori_image_path = os.path.join(image_path, filename)
                    new_image_path = os.path.join('/data/zhr/spc_vehicle_small/image/val', new_filename)
                shutil.copy(ori_image_path, new_image_path)
                
                # new_data_infos.append(data_infos_temp)

                # # save image
                # ori_image_path = os.path.join(image_path, filename)
                # new_image_path = os.path.join(img_save_path, new_filename)
                # shutil.copy(ori_image_path, new_image_path)

            # import pdb; pdb.set_trace()

    temp_out_train = '/home/zhr/mmdet-rfla/tools/dataset_converters/new_temp_train.json'
    temp_out_val = '/home/zhr/mmdet-rfla/tools/dataset_converters/new_temp_val.json'
    mmcv.dump(new_data_infos_train, temp_out_train)
    mmcv.dump(new_data_infos_val, temp_out_val)
    new_vehicle_train = mmcv.load(temp_out_train)
    new_vehicle_val = mmcv.load(temp_out_val)

    new_vehicle_train_dict = dict()
    new_vehicle_train_dict['categories'] = [{"id": 0, "name": "vehicle", "supercategory": "mark"}]
    new_vehicle_val_dict = dict()
    new_vehicle_val_dict['categories'] = [{"id": 0, "name": "vehicle", "supercategory": "mark"}]
    '''
    info{
    "year" : int,                # 年份
    "version" : str,             # 版本
    "description" : str,         # 详细描述信息
    "contributor" : str,         # 作者
    "url" : str,                 # 协议链接
    "date_created" : datetime,   # 生成日期
    "noisiness"  :  str          # 噪声含量
    "mode"  : str                # 噪声模式
    }
    '''
    new_vehicle_train_dict['info'] = {}
    new_vehicle_train_dict['info']['year'] = 2024
    new_vehicle_train_dict['info']['version'] = '1.0'
    new_vehicle_train_dict['info']['description'] = "This is a vehicle dataset"
    new_vehicle_train_dict['info']['contributor'] = 'Zhu Haoran'
    new_vehicle_train_dict['info']['url'] = 'null'
    new_vehicle_train_dict['info']['data_created'] = '2024.3.11'
    new_vehicle_train_dict['info']['datasets'] = 'DIOR/AI-TOD-v2/DOTA-v2'
    new_vehicle_train_dict['annotations'] = []  ####需添加
    new_vehicle_train_dict['images'] = []   ##3需添加
    targetidssp1 = 0
    for j in range(0, len(new_vehicle_train)):
        Anns = new_vehicle_train[j]['ann']
        segmentations = []
        areas = []
        image_item = dict()
        image_item['file_name'] = new_vehicle_train[j]['file_name']
        image_item['height'] = new_vehicle_train[j]['height']
        image_item['width'] = new_vehicle_train[j]['width']
        image_item['id'] = new_vehicle_train[j]['id']
        new_vehicle_train_dict['images'].append(image_item)
        for k in range(0, len(Anns['labels'])):
            annotation_item = dict()
            annotation_item['image_id'] = new_vehicle_train[j]['id']
            annotation_item['bbox'] = [round(Anns['bboxes'][k][0], 1), round(Anns['bboxes'][k][1], 1), round(Anns['bboxes'][k][2], 1), round(Anns['bboxes'][k][3], 1)]
            annotation_item['area'] = round(Anns['bboxes'][k][2] * Anns['bboxes'][k][3], 1)
            annotation_item['segmentation'] = [[Anns['bboxes'][k][0], Anns['bboxes'][k][1], Anns['bboxes'][k][0] + Anns['bboxes'][k][2], Anns['bboxes'][k][1] + Anns['bboxes'][k][3]]]
            annotation_item['category_id'] = Anns['labels'][k]
            annotation_item['iscrowd'] = Anns['iscrowd'][k]
            annotation_item['ignore'] = Anns['ignore'][k]
            annotation_item['new_category_name'] = Anns['new_category_name'][k]
            annotation_item['ori_category_name'] = Anns['ori_category_name'][k]
            annotation_item['id'] = targetidssp1 # 从0开始
            targetidssp1 = targetidssp1 + 1
            new_vehicle_train_dict['annotations'].append(annotation_item)
    anno_prefix = '/data/zhr/spc_vehicle_small/annotations/'
    mkdir(anno_prefix)
    out_ann_file = anno_prefix + 'train.json'
    print("当前数据集共有", targetidssp1, '个标签')
    print('save to {}'.format(out_ann_file))
    json.dump(new_vehicle_train_dict, open(out_ann_file, 'w'), indent = 4) # indent=4 更加美观显示 ##########################

    new_vehicle_val_dict['info'] = {}
    new_vehicle_val_dict['info']['year'] = 2024
    new_vehicle_val_dict['info']['version'] = '1.0'
    new_vehicle_val_dict['info']['description'] = "This is a vehicle dataset"
    new_vehicle_val_dict['info']['contributor'] = 'Zhu Haoran'
    new_vehicle_val_dict['info']['url'] = 'null'
    new_vehicle_val_dict['info']['data_created'] = '2024.1.28'
    new_vehicle_val_dict['info']['datasets'] = 'DIOR/AI-TOD-v2/DOTA-v2/XVIEW'
    new_vehicle_val_dict['annotations'] = []  ####需添加
    new_vehicle_val_dict['images'] = []   ##3需添加
    targetidssp1 = 0
    for j in range(0, len(new_vehicle_val)):
        Anns = new_vehicle_val[j]['ann']
        segmentations = []
        areas = []
        image_item = dict()
        image_item['file_name'] = new_vehicle_val[j]['file_name']
        image_item['height'] = new_vehicle_val[j]['height']
        image_item['width'] = new_vehicle_val[j]['width']
        image_item['id'] = new_vehicle_val[j]['id']
        new_vehicle_val_dict['images'].append(image_item)
        for k in range(0, len(Anns['labels'])):
            annotation_item = dict()
            annotation_item['image_id'] = new_vehicle_val[j]['id']
            annotation_item['bbox'] = [round(Anns['bboxes'][k][0], 1), round(Anns['bboxes'][k][1], 1), round(Anns['bboxes'][k][2], 1), round(Anns['bboxes'][k][3], 1)]
            annotation_item['area'] = round(Anns['bboxes'][k][2] * Anns['bboxes'][k][3], 1)
            annotation_item['segmentation'] = [[Anns['bboxes'][k][0], Anns['bboxes'][k][1], Anns['bboxes'][k][0] + Anns['bboxes'][k][2], Anns['bboxes'][k][1] + Anns['bboxes'][k][3]]]
            annotation_item['category_id'] = Anns['labels'][k]
            annotation_item['iscrowd'] = Anns['iscrowd'][k]
            annotation_item['ignore'] = Anns['ignore'][k]
            annotation_item['new_category_name'] = Anns['new_category_name'][k]
            annotation_item['ori_category_name'] = Anns['ori_category_name'][k]
            annotation_item['id'] = targetidssp1 # 从0开始
            targetidssp1 = targetidssp1 + 1
            new_vehicle_val_dict['annotations'].append(annotation_item)
    anno_prefix = '/data/zhr/spc_vehicle_small/annotations/'
    mkdir(anno_prefix)
    out_ann_file = anno_prefix + 'val.json'
    print("当前数据集共有", targetidssp1, '个标签')
    print('save to {}'.format(out_ann_file))
    json.dump(new_vehicle_val_dict, open(out_ann_file, 'w'), indent = 4) # indent=4 更加美观显示 ##########################


    print('max_width:', max_width)
    print('min_width:', min_width)
    print('max_height:', max_height)
    print('min_height:', min_height)

# Provide the path to the folder containing the COCO JSON files
folder_path = '123'
merge_coco_files(folder_path)


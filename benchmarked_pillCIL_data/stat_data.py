from calendar import c
import os
import pdb
import numpy as np 
import os.path as osp
import cv2
import copy
import json

def stat_data(path_data_ori, mode):

    # source
    ori_img_path = osp.join(path_data_ori, 'images')
    ori_label_path = osp.join(path_data_ori, 'labels')


    all_json_names = os.listdir(ori_label_path)

    # first get the list of all label
    all_categories_list = []
    for json_name in all_json_names:
        full_json_name = osp.join(ori_label_path, json_name)
        
        # read json data
        with open(full_json_name, 'r') as fp:
            json_data = json.load(fp)

        for box in json_data['boxes']:
            label = box['label']
            if label not in all_categories_list:
                all_categories_list.append(label)

    all_categories_list = sorted(all_categories_list)

    json_stat = {"min_num_samples": 1000000000,
                "max_num_samples": -1,
                "average_num_samples":-1,
                "stat_samples_per_class":{}}
    for json_name in all_json_names:
        full_json_name = osp.join(ori_label_path, json_name)
        
        # read json data
        with open(full_json_name, 'r') as fp:
            json_data = json.load(fp)

        img_name = json_data['path']
        boxes_info = json_data['boxes']

        # test box coordinates
        for index, box in enumerate(boxes_info):
            xmin, xmax, ymin, ymax = box["x"], box["x"] + box["w"],\
                                        box["y"], box["y"] + box["h"]
            
            label = box['label']
            if label not in json_stat["stat_samples_per_class"]:
                json_stat["stat_samples_per_class"][label] = 0
            json_stat["stat_samples_per_class"][label] += 1  

    # fill min, max, average stat numbers
    average_sam = 0
    for each_class in json_stat["stat_samples_per_class"]:
        num_sam = json_stat["stat_samples_per_class"][each_class]

        json_stat["min_num_samples"] = min(json_stat["min_num_samples"], num_sam)
        json_stat["max_num_samples"] = max(json_stat["max_num_samples"], num_sam)


        average_sam += num_sam

    average_sam /= len(json_stat["stat_samples_per_class"].keys())
    json_stat["average_num_samples"] = average_sam
    
    # write to txt file
    with open(mode+".json", 'w') as fp:
        json.dump(json_stat, fp, indent=4, sort_keys=True)
        


if __name__ == '__main__':
    path_train = '/home/tung/Tung/research/full_data_pill/train'
    path_test = '/home/tung/Tung/research/full_data_pill/test'

    
    stat_data(path_train, 'train')
    stat_data(path_test, 'test')

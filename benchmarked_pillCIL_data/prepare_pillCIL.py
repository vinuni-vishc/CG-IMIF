from calendar import c
from tqdm import tqdm
import os
import pdb
import numpy as np 
import os.path as osp
import cv2
import copy
import json

des_path = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Large'
if os.path.isdir(des_path) is False:
    os.mkdir(des_path)

def prepare_data_CIL_pill(path_data_ori, mode):
    if os.path.isdir(osp.join(des_path, mode)) is False:
        os.mkdir(osp.join(des_path, mode))

    # source
    ori_img_path = osp.join(path_data_ori, 'images')
    ori_label_path = osp.join(path_data_ori, 'labels')

    # des
    des_img_path = osp.join(des_path, mode)
    des_label_path = osp.join(des_path, mode+'.txt')

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
            label = label.replace(' ', '*')
            if label not in all_categories_list:
                all_categories_list.append(label)

    all_categories_list = sorted(all_categories_list)
    # now write categories list to json
    with open("./all_cat_pill_large.json", 'w') as fp:
        json.dump({"all_cat": all_categories_list}, fp, indent = 4)


    str_item_txt = ''
    for json_name in tqdm(all_json_names):
        full_json_name = osp.join(ori_label_path, json_name)
        
        # read json data
        with open(full_json_name, 'r') as fp:
            json_data = json.load(fp)

        img_name = json_data['path']
        boxes_info = json_data['boxes']

        full_img_name = osp.join(ori_img_path, img_name)
        # read img 
        img = cv2.imread(full_img_name)
        tmp_img = copy.copy(img)
        
        # test box coordinates
        for index, box in enumerate(boxes_info):
            xmin, xmax, ymin, ymax = box["x"], box["x"] + box["w"],\
                                        box["y"], box["y"] + box["h"]
            
            label = box['label']
            label = label.replace(' ', '*')

            single_pill_crop = tmp_img[ymin:ymax,xmin:xmax]

            full_name_img_write = label + "_" + img_name.split(".")[0] + "-" +str(index) +".jpg"
            full_path_img_write = osp.join(des_img_path, full_name_img_write)
            cv2.imwrite(full_path_img_write, single_pill_crop)

            # write to txt
            str_item_txt += full_path_img_write + " " + str(all_categories_list.index(label)) + '\n'           
        
    
    # write to txt file
    with open(des_label_path, 'w') as fp:
        fp.write(str_item_txt)
        


if __name__ == '__main__':
    path_train = '/home/tung/Tung/research/full_data_pill/train'
    path_test = '/home/tung/Tung/research/full_data_pill/test'

    # prepare train
    prepare_data_CIL_pill(path_train, 'train')

    # # prepare test
    prepare_data_CIL_pill(path_test, 'test')

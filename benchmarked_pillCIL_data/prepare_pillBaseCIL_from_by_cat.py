from calendar import c
from tqdm import tqdm
import os
import pdb
import numpy as np 
import os.path as osp
import cv2
import copy
import pdb
import json

alpha = 0
beta = 120

# des_path_rgb = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base_X'
# des_path_edge = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base_Edge'
des_path_texture = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base_Texture'
# if os.path.isdir(des_path_rgb) is False:
#     os.mkdir(des_path_rgb)

# if os.path.isdir(des_path_edge) is False:
#     os.mkdir(des_path_edge)

if os.path.isdir(des_path_texture) is False:
    os.mkdir(des_path_texture)

def find_dropped_cat(path_json_by_cat_train):
    '''
        Find category that does not satisfy constraint:
            + Constraint 1: # of samples does not exceed 
            + Constraint 2: # of samples are not smaller than ...
    '''
    with open(path_json_by_cat_train, 'r') as fp:
        by_cat_data = json.load(fp)

    list_ok_categories = []

    for cat_name in by_cat_data.keys():
        num_samples_cat = len(by_cat_data[cat_name])

        is_ok = False
        if num_samples_cat > alpha and num_samples_cat < beta:
            is_ok = True
        
        if is_ok:
            list_ok_categories.append(cat_name)

    return list_ok_categories

def stat_data(path_json_by_cat, list_ok_categories, mode):
    with open(path_json_by_cat, 'r') as fp:
        by_cat_data = json.load(fp)

    # also perform stat on data
    json_stat = {"min_num_samples": 1000000000,
                "max_num_samples": -1,
                "average_num_samples":-1,
                "num_categories": 0,
                "stat_samples_per_class":{}}

    total_num_samples = 0
    for cat in by_cat_data:
        if cat not in list_ok_categories:
            continue
        num_samples = len(by_cat_data[cat])
        total_num_samples += num_samples
        json_stat["min_num_samples"] = min(json_stat["min_num_samples"], num_samples)
        json_stat["max_num_samples"] = max(json_stat["max_num_samples"], num_samples)

        json_stat['stat_samples_per_class'][cat] = num_samples
        json_stat['num_categories'] += 1

    json_stat['stat_samples_per_class'] = dict(sorted(json_stat['stat_samples_per_class'].items(), key=lambda item: item[1]))
    json_stat["average_num_samples"] = total_num_samples/json_stat['num_categories']

    with open(mode+'_base_stat.json', 'w') as fp:
        json.dump(json_stat, fp, indent=4)



def prepare_data_CIL_pill(path_json_by_cat, mode, list_ok_categories):
    # if os.path.isdir(osp.join(des_path_rgb, mode)) is False:
    #     os.mkdir(osp.join(des_path_rgb, mode))

    # if os.path.isdir(osp.join(des_path_edge, mode)) is False:
    #     os.mkdir(osp.join(des_path_edge, mode))

    if os.path.isdir(osp.join(des_path_texture, mode)) is False:
        os.mkdir(osp.join(des_path_texture, mode))

    with open(path_json_by_cat, 'r') as fp:
        by_cat_json_data = json.load(fp)

    # des rgb
    # des_img_path_rgb = osp.join(des_path_rgb, mode)
    # des_label_path_rgb = osp.join(des_path_rgb, mode+'.txt')

    # des_img_path_edge = osp.join(des_path_edge, mode)
    # des_label_path_edge = osp.join(des_path_edge, mode+'.txt')

    des_img_path_texture = osp.join(des_path_texture, mode)
    des_label_path_texture = osp.join(des_path_texture, mode+'.txt')


    # all_categories_list = sorted(all_categories_list)
    # # now write categories list to json
    # with open("./all_cat_pill_large.json", 'w') as fp:
    #     json.dump({"all_cat": all_categories_list}, fp, indent = 4)

    str_item_txt_rgb = ''
    str_item_txt_edge = ''
    str_item_txt_texture = ''

    for cat in tqdm(by_cat_json_data):
        if cat not in list_ok_categories:
            continue
        
        all_samples_cat = by_cat_json_data[cat]
        for idx, sample_cat in enumerate(all_samples_cat):
            path_img = sample_cat['path_img']
            bbox = sample_cat['bbox']
            [xmin, xmax, ymin, ymax] = bbox

            # img = cv2.imread(path_img)
            # single_pill = img[ymin:ymax, xmin:xmax]
            full_name_img_write = cat + "_" + path_img.split("/")[-1].split('.')[0] + "-"+str(idx)+".jpg"
            
            # full_path_img_write_rgb = osp.join(des_img_path_rgb, full_name_img_write)
            # full_path_img_write_edge = osp.join(des_img_path_edge, full_name_img_write)
            full_path_img_write_texture = osp.join(des_img_path_texture, full_name_img_write)

            # write rgb
            # cv2.imwrite(full_path_img_write_rgb, single_pill)
            # str_item_txt_rgb += full_path_img_write_rgb + " " + str(list_ok_categories.index(cat)) + '\n'

            # img_gray = cv2.cvtColor(single_pill, cv2.COLOR_BGR2GRAY)
            # img_blur = cv2.GaussianBlur(img_gray, (9,9), 0)

            # edge_img = cv2.Canny(image=img_blur, threshold1=20, threshold2=50)
            
            # # write edge 
            # # cv2.imwrite(full_path_img_write_edge, edge_img)
            # # str_item_txt_edge += full_path_img_write_edge + " " + str(list_ok_categories.index(cat)) + '\n'

            # # write texture
            # cv2.imwrite(full_path_img_write_texture, img_gray-img_blur)
            str_item_txt_texture += full_path_img_write_texture + " " + str(list_ok_categories.index(cat)) + '\n'


    # write to txt file
    # with open(des_label_path_rgb, 'w') as fp:
    #     fp.write(str_item_txt_rgb)

    # with open(des_label_path_edge, 'w') as fp:
    #     fp.write(str_item_txt_edge)

    with open(des_label_path_texture, 'w') as fp:
        fp.write(str_item_txt_texture)
        


if __name__ == '__main__':
    path_by_cat_train = '/home/tung/Tung/research/Open-Pill/FACIL/benchmarked_pillCIL_data/by_category_train.json'
    path_by_cat_test = '/home/tung/Tung/research/Open-Pill/FACIL/benchmarked_pillCIL_data/by_category_test.json'
    
    # find all satisfied categories
    list_ok_categories = find_dropped_cat(path_by_cat_train)
    list_ok_categories = sorted(list_ok_categories)
    # now dump list ok to json list
    with open('./all_cat_pill_base.json', 'w') as fp:
        json.dump({"all_cat":list_ok_categories}, fp, indent=4)

    stat_data(path_by_cat_train, list_ok_categories, 'train')
    stat_data(path_by_cat_test, list_ok_categories, 'test')

    # prepare train
    prepare_data_CIL_pill(path_by_cat_train, 'train', list_ok_categories)

    # # prepare test
    prepare_data_CIL_pill(path_by_cat_test, 'test', list_ok_categories)

    # then dump to category list

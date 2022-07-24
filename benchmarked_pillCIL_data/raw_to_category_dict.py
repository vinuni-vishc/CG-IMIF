'''
    This script helps convert raw json data into category json
'''
import os.path as osp
import json
import os

from tqdm import tqdm

def to_categorical_json(path_data_ori, mode):

    # source
    ori_img_path = osp.join(path_data_ori, 'images')
    ori_label_path = osp.join(path_data_ori, 'labels')


    all_json_names = os.listdir(ori_label_path)
    json_category_save = {}

    for json_name in tqdm(all_json_names):
        full_json_name = osp.join(ori_label_path, json_name)
        
        # read json data
        with open(full_json_name, 'r') as fp:
            json_data = json.load(fp)

        img_name = json_data['path']
        boxes_info = json_data['boxes']

        full_img_name = osp.join(ori_img_path, img_name)
        
        # test box coordinates
        for index, box in enumerate(boxes_info):
            xmin, xmax, ymin, ymax = box["x"], box["x"] + box["w"],\
                                        box["y"], box["y"] + box["h"]
            
            label = box['label']
            label = label.replace(' ', '*')

            if label not in json_category_save:
                json_category_save[label] = []

            instance_add = {}
            instance_add['path_img'] = full_img_name
            instance_add['bbox'] = [xmin, xmax, ymin, ymax]

            json_category_save[label].append(instance_add)
    
    out_json_name = "by_category_"+mode+".json"

    with open(out_json_name, 'w') as fp:
        json.dump(json_category_save, fp, indent=4)

if __name__ == '__main__':
    path_train = '/home/tung/Tung/research/full_data_pill/train'
    path_test = '/home/tung/Tung/research/full_data_pill/test'

    # prepare train
    to_categorical_json(path_train, 'train')

    # # prepare test
    to_categorical_json(path_test, 'test')
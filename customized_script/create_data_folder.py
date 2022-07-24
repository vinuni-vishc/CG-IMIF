import os
import json
import shutil

# from tqdm import tqdm

base_path_tool = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base'

if os.path.isdir(base_path_tool) is False:
    os.mkdir(base_path_tool)

train_path_tool = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base/train'

if os.path.isdir(train_path_tool) is False:
    os.mkdir(train_path_tool)

test_path_tool = '/home/tung/Tung/research/Open-Pill/FACIL/data/Pill_Base/test'

if os.path.isdir(test_path_tool) is False:
    os.mkdir(test_path_tool)

def create_customized_data(path_json, mode, list_cat_idx=None):
    # list_cat_idx is default for train but should be specified for test

    if mode == 'train':
        data_folder_path = train_path_tool
    else:
        data_folder_path = test_path_tool

    path_txt_file = os.path.join(base_path_tool, mode+'.txt')
    # create train txt and train folder image

    str_item_txt = ''
    with open(path_json, 'r') as fp:
        json_data = json.load(fp)

    if mode == 'train':
        list_cat_idx = [*json_data]

    for cat in json_data:
        for img_cat in json_data[cat]:
            img_name = img_cat.split('/')[-1]

            src_img_path = img_cat
            des_img_path = os.path.join(data_folder_path, img_name)
            des_img_path = des_img_path.replace(' ', '__')

            shutil.copy(src_img_path, des_img_path)

            # prepare txt file

            str_txt = des_img_path + " " + str(list_cat_idx.index(cat)) + '\n'
            str_item_txt += str_txt

    with open(path_txt_file, 'w') as fp:
        fp.write(str_item_txt)

    return list_cat_idx


if __name__ == '__main__':
    path_json_train = '/home/tung/Tung/research/Open-Pill/my_dataset/json/train_new.json'
    path_json_test = '/home/tung/Tung/research/Open-Pill/my_dataset/json/test_new.json'

    list_cat_idx = create_customized_data(path_json_train, 'train')
    list_cat_idx = create_customized_data(path_json_test, 'test', list_cat_idx)


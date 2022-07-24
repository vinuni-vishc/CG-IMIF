import os
import json
import copy

def split_data(path_train_ori, path_test_ori, 
                path_train_new_split, path_test_new_split):

    with open(path_train_ori, 'r') as fp:
        json_ori_train = json.load(fp)

    with open(path_test_ori, 'r') as fp:
        json_ori_test = json.load(fp)

    json_new_split_train = copy.copy(json_ori_train)
    json_new_split_test = copy.copy(json_ori_test)

    # iterate through cat in train and check exist in test
    for cat in json_new_split_train.keys():
        if cat not in json_new_split_test:
            # remove two samples from train to add it to test

            json_new_split_test[cat] = json_new_split_train[cat][-2:]
            json_new_split_train[cat] = json_new_split_train[cat][:-2]

    with open(path_train_new_split, 'w') as fp:
        json.dump(json_new_split_train, fp, indent=4)

    with open(path_test_new_split, 'w') as fp:
        json.dump(json_new_split_test, fp, indent=4)
    

if __name__ == '__main__':
    path_train_ori = '/home/tung/Tung/research/Open-Pill/my_dataset/json/train.json'
    path_test_ori = '/home/tung/Tung/research/Open-Pill/my_dataset/json/test.json'
    path_train_new_split = '/home/tung/Tung/research/Open-Pill/my_dataset/json/train_new.json'
    path_test_new_split = '/home/tung/Tung/research/Open-Pill/my_dataset/json/test_new.json'

    split_data(path_train_ori, path_test_ori, path_train_new_split, path_test_new_split)
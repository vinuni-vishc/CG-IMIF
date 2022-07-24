import json
import os 
import pdb
import math

# specify param to build periodical data here
class_per_period = 5

def prepare_period_data(path_data_by_cat_train, path_data_by_cat_test,
                        path_data_period_folder):

    # read json cat data train
    with open(path_data_by_cat_train, 'r') as fp:
        all_cat_data_train = json.load(fp)

    json_period_train = {}
    n_classes = len(all_cat_data_train.keys())
    all_classes_train = [*all_cat_data_train]

    num_period = math.ceil(n_classes / class_per_period)

    for id_period in range(num_period):
        json_period_train[id_period] = {}

        run_class_period = class_per_period
        if id_period == num_period - 1:
            run_class_period = n_classes - id_period*class_per_period
        for idx_class in range(run_class_period):
            convert_idx_class = id_period*class_per_period + idx_class 

            class_name = all_classes_train[convert_idx_class]
            json_period_train[id_period][class_name] = all_cat_data_train[class_name]

    path_data_period_folder = os.path.join(path_data_period_folder, str(class_per_period)+"_classes")

    if os.path.isdir(path_data_period_folder) is False:
        os.mkdir(path_data_period_folder)

    # write train
    train_path_period = os.path.join(path_data_period_folder, 'period_train.json')
    with open(train_path_period, 'w') as fp:
        json.dump(json_period_train, fp, indent=4)

    # from info of train, extract test
    json_period_test = {}

    # read json cat data test
    with open(path_data_by_cat_test, 'r') as fp:
        all_cat_data_test = json.load(fp)
    

    all_classes_test = [*all_cat_data_test]

    if len(all_cat_data_test) != len(all_cat_data_train):
        print('error')

    for id_period in range(num_period):
        json_period_test[id_period] = {}

        run_class_period = class_per_period
        if id_period == num_period - 1:
            run_class_period = n_classes - id_period*class_per_period
        for idx_class in range(run_class_period):
            convert_idx_class = id_period*class_per_period + idx_class 

            class_name = all_classes_train[convert_idx_class] # careful this should be train since it refers to train category order
            json_period_test[id_period][class_name] = all_cat_data_test[class_name]

    # write test
    test_path_period = os.path.join(path_data_period_folder, 'period_test.json')
    with open(test_path_period, 'w') as fp:
        json.dump(json_period_test, fp, indent=4)


if __name__ == '__main__':
    path_data_by_cat_train = '/home/tung/Tung/research/Open-Pill/my_dataset/json/train_new.json'
    path_data_by_cat_test = '/home/tung/Tung/research/Open-Pill/my_dataset/json/test_new.json'
    path_data_period = '/home/tung/Tung/research/Open-Pill/period_data/'


    prepare_period_data(path_data_by_cat_train, path_data_by_cat_test, path_data_period)
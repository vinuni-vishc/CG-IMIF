import json
import pdb
import random

def generate_class_order(path_json_class):
    with open(path_json_class, 'r') as fp:
        json_data = json.load(fp)

    num_categories = len(json_data['all_cat'])
    class_order = [*range(num_categories)]

    # random.shuffle(class_order)
    pdb.set_trace()

    

if __name__ == '__main__':
    path_json_class = '/home/tung/Tung/research/Open-Pill/FACIL/benchmarked_pillCIL_data/all_cat_pill_base.json'
    
    generate_class_order(path_json_class)
import pdb
import re

def cal_avg(input_str_acc):
    list_acc = list(map(float, re.split(r'\s+', input_str_acc)))
    print(sum(list_acc)/len(list_acc))

if __name__ == "__main__":
    string_acc = '29.6 30.0 28.6 38.5 37.8 46.9 47.5 38.8 39.0 50.2 45.8 44.4 46.0 45.0'
    cal_avg(string_acc)
'''
    This script is specially designed to draw forg in incremental learning
'''

import pandas as pd
import pdb
import numpy as np
import re
import matplotlib.pyplot as plt

num_tasks = 15
type_file = '.pdf'
choose_fontsize = 15

# # Numtask=5

if num_tasks == 5:
    task_id_list = [1, 2, 3, 4, 5]

    eeil_base = '0 48.6 46.6  54.4 49.7'
    eeil_ours = '0 45.9 44.2 50.2 46.4'

    bic_base = '0 8.7 8.3 32.4 30.8'
    bic_ours = '0 11.4 0.7 9.3 9.6'

    lucir_base = '0 34.2 40.9 50.8 50.6'
    lucir_ours = '0 22.5 30.5 39.5 40.1'

'''************************************************'''

# Numtask=10
if num_tasks == 10:
    task_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    eeil_base = '0 42.2 40.8  39.0 50.5 47.5  48.3 48.2 43.8 48.8'
    eeil_ours = '0 45.2 42.3 42.2  40.5 44.6 45.4 49.8 44.6 47.2'

    bic_base = '0 25.9 19.8 55.7 32.4 25.0 19.7 29.1 25.0 41.9'
    bic_ours = '0 25.9 24.6 22.2 25.1 25.2  22.3 18.9 17.4 16.5'

    lucir_base = '0 36.1 34.6 43.6 48.2 47.8 48.1 48.4 46.3 45.8'
    lucir_ours = '0 33.1 36.6 35.4 38.1 37.9 37.8 41.6 39.0 41.4'

# '''***********************************************'''

# Numtask=15
if num_tasks==15:
    task_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    eeil_base = '0 65.2 49.2 46.8 51.6 47.2 50.7 41.8 41.3 45.0  52.0 49.7 44.0 46.6  44.7'
    eeil_ours = '0 56.5 57.6 38.1 43.8 40.4 45.5 42.4 40.9 41.8 47.9 47.5 48.9 45.1 50.9'

    bic_base = '0 24.3 23.4 26.6 39.8 27.7 25.1 25.8 22.9 24.1 26.7 29.2 27.3 28.5 25.6'
    bic_ours = '0 27.0 30.5 22.5 47.4 31.1 18.8 19.6 16.7 37.2 30.9  26.1 22.0  19.5 33.6'

    lucir_base = '0 48.7 42.7 45.1 51.7 46.5 47.8 47.3 45.4 46.5 46.9 48.2 46.6 49.0 47.1'
    lucir_ours = '0 32.2 32.9 32.5 37.7 34.9 42.1 38.6 38.1 40.4 44.8 48.3 44.0 46.5 44.0'

#######Transform to list of number###########

eeil_base = list(map(float, re.split(r'\s+', eeil_base)))
eeil_ours = list(map(float, re.split(r'\s+', eeil_ours)))

bic_base = list(map(float, re.split(r'\s+', bic_base)))
bic_ours = list(map(float, re.split(r'\s+', bic_ours)))

lucir_base = list(map(float, re.split(r'\s+', lucir_base)))
lucir_ours = list(map(float, re.split(r'\s+', lucir_ours)))


######## Multiply ########
eeil_base = list(np.array(eeil_base))
eeil_ours = list(np.array(eeil_ours))

bic_base = list(np.array(bic_base))
bic_ours = list(np.array(bic_ours))

lucir_base = list(np.array(lucir_base))
lucir_ours = list(np.array(lucir_ours))

plt.plot(task_id_list, eeil_base,   label = 'EEIL-Base', marker='o', color='r', linestyle='--', linewidth=2)
plt.plot(task_id_list, eeil_ours,   label = 'EEIL-IMIF (Ours)',  marker='o', color='r', linewidth=2)
plt.plot(task_id_list, bic_base, label = 'BiC-Base', marker='x', color='green', linestyle='--', linewidth=2)
plt.plot(task_id_list, bic_ours, label = 'BiC-IMIF (Ours)', marker='x', color='green', linewidth=2)
plt.plot(task_id_list, lucir_base, label = 'LUCIR-Base', marker='v', color='orange', linestyle='--', linewidth=2)
plt.plot(task_id_list, lucir_ours, label = 'LUCIR-IMIF (Ours)', marker='v', color='orange', linewidth=2)

plt.xlabel('TaskID', fontsize=choose_fontsize)
plt.ylabel('Forgetting Rate', fontsize=choose_fontsize)
# plt.legend(loc='upper right')
plt.legends = []

plt.xticks(task_id_list, fontsize=choose_fontsize)
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=choose_fontsize)
plt.title(str(len(task_id_list))+' tasks', fontsize=choose_fontsize)
plt.savefig('paper_forg_acc_'+str(len(task_id_list))+'_tasks'+type_file)


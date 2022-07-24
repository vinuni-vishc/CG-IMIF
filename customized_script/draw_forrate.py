'''
    This script is specially designed to draw accuracy in incremental learning
'''

import pandas as pd
import matplotlib.pyplot as plt


# Numtask=4
# learning_subtasks = [19, 38, 57, 76]
# scores_eeil = [0.289474, 0.105263, 0.078947, 0.0000]
# scores_finetuning = [0.868421, 0.526316, 0.421053, 0.00000]
# scores_joint_training = [0.263158, 0.052632, 0.0000, 0.00000]

# Numtask=7
# learning_subtasks = [11, 22, 33, 44, 55, 66, 76]
# scores_eeil = [0.136364, 0.363636, 0.227273, 0.409091, 0.136364, 0.272727, 0.000]
# scores_finetuning = [0.500000, 0.363636, 0.500000, 0.681818, 0.227273, 0.590909, 0.00000]
# scores_joint_training = [0.090909, 0.045455, 0.045455, 0.045455, 0.000, 0.0000, 0.000]

# Numtask = 12
learning_subtasks = [7, 14, 21, 28, 34, 40, 46, 52, 58, 64, 70, 76]
scores_eeil = [1.0000, 0.785714, 0.857143, 0.857134, 1.00000, 1.0000, 0.750000, 1.0000, 0.166667, 0.0000, 0.0000, 0.00000]
scores_finetuning = [1.00000, 0.285714, 0.285714, 0.714286, 0.666667, 0.416667, 0.500000, 0.833333, 0.666667, 0.750000, 0.083333, 0.0000]
scores_joint_training = [0.285714, 0.0000, 0.285713, 0.285714, 0.083333, 0.250000, 0.166667, 0.0000, 0.0000, 0.0000, 0.166667, 0.0000]

df = pd.DataFrame(list(zip( scores_eeil , scores_finetuning, scores_joint_training )),
                  index = learning_subtasks, 
                  columns = ['Baseline', 'Fine-Tuning', 'Joint-Training'])

ax = df.plot(title='Forgetting rate on 12 tasks CIL')
ax.set(xlabel='Task-ID', ylabel='Forgetting Rate')
ax.figure.savefig('results_plot_forg_rate_12_tasks.png')


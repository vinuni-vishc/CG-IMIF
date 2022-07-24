'''
    This script is specially designed to draw accuracy in incremental learning
'''

import pandas as pd
import matplotlib.pyplot as plt


# Numtask=4
# learning_subtasks = [19, 38, 57, 76]
# scores_eeil = [0.894737, 0.723684, 0.728070, 0.717105]
# scores_finetuning = [0.894737, 0.447368, 0.298246, 0.138158]
# scores_joint_training = [0.894737, 0.736842, 0.763158, 0.723684]

# Numtask=7
# learning_subtasks = [11, 22, 33, 44, 55, 66, 76]
# scores_eeil = [0.500000, 0.727273, 0.545455, 0.602273, 0.636364, 0.628788, 0.505844]
# scores_finetuning = [0.500000, 0.409091, 0.196970, 0.193182, 0.181818, 0.174242, 0.086364]
# scores_joint_training = [0.500000, 0.409091, 0.590909, 0.715909, 0.727273, 0.757576, 0.729221]

# Numtask = 12
learning_subtasks = [7, 14, 21, 28, 34, 40, 46, 52, 58, 64, 70, 76]
scores_eeil = [1.000000, 0.821429, 0.833333, 0.750000, 0.814286, 0.680556, 0.676871, 0.735119, 0.097884, 0.016667, 0.022727, 0.013889]
scores_finetuning = [1.000000, 0.392857, 0.238095, 0.214286, 0.219048, 0.176587, 0.112245, 0.122024, 0.117725, 0.083333, 0.060606, 0.090278]
scores_joint_training = [1.000000, 0.928571, 0.880952, 0.892857, 0.900000, 0.916667, 0.916667, 0.927083, 0.907407, 0.925000, 0.945887, 0.944444]

df = pd.DataFrame(list(zip( scores_eeil , scores_finetuning, scores_joint_training )),
                  index = learning_subtasks, 
                  columns = ['Baseline', 'Fine-Tuning', 'Joint-Training'])

ax = df.plot(title='Top 1 Accuracy on 12 tasks CIL')
ax.set(xlabel='Num classes', ylabel='Top-1-Accuracy')
ax.figure.savefig('results_plot.png')


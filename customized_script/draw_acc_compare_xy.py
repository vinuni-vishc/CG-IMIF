'''
    This script is specially designed to draw accuracy in incremental learning
'''

import pandas as pd
import matplotlib.pyplot as plt


# Numtask=5
# learning_subtasks = [20, 40, 60, 79, 98]
# eeil_edge_early = [0.902516,0.819002,0.696368,0.667872,0.628280]
# eeil_edge_intermediate = [0.911950,0.814219,0.775332,0.763019,0.731925]
# eeil_histo_early = [0.933962,0.831250,0.755838,0.743875,0.743852]
# eeil_histo_intermediate = [0.953, 0.864, 0.793, 0.780, 0.750]

# Numtask=10
# learning_subtasks = [10, 20, 30, 40, 50, 60, 70, 80, 89, 98]
# eeil_edge_early = [1.000000,0.807058,0.785819,0.706379,0.673130,0.663984,0.622863,	0.593867,0.034714,0.347779]
# eeil_edge_intermediate = [0.993789,0.835166,0.796135,0.745283,0.707941,0.682070,0.657986,0.609029,0.591457,0.609201] 
# eeil_histo_early = [0.993789,0.828401,0.681386,0.714240,0.730022,0.747428,0.754909,0.744496,0.733012,0.744165]
# eeil_histo_intermediate = [0.987578,0.866064,0.776552,0.772575,0.784182,0.770409,0.774215,0.767366	,0.714723,0.742302]

# # Numtask = 12
learning_subtasks = [7, 14, 21, 28, 35, 42, 49, 56, 62, 68, 74, 80, 86, 92, 98]
eeil_edge_early = [1.000000,0.895652,0.767476,0.709947,0.711886,0.697147,0.623774,	0.622071,0.103842,0.100258,0.205636,0.472914,0.524991,0.611134,0.638669]
eeil_edge_intermediate = [1.000000,0.869196,0.818630,0.790617,0.727538,0.711685,0.678173,0.668124,0.636774,0.646812,0.611314,0.636396,0.616084,0.603139,0.594542]
eeil_histo_early = [1.000000,0.869565,0.846398,0.819409,0.855490,0.836037,0.823089,0.791923,0.736627,0.766376,0.771543,0.746359,0.707289,0.721382,0.732087] 
eeil_histo_intermediate = [1.000000,0.943478,0.882968,0.851051,0.863293,0.828117,0.819057,0.799822,0.753624	,0.771862,0.739318,0.762233,0.715227,0.690976,0.699943]



df = pd.DataFrame(list(zip(eeil_edge_early, eeil_edge_intermediate, eeil_histo_early, eeil_histo_intermediate)),
                  index = learning_subtasks,
                  columns = ['EEIL + Stream Edge + Early Fusion', 
                  'EEIL + Stream Edge + Intermediate Fusion', 
                  'EEIL + Stream Histogram + Early Fusion',
                  'EEIL + Stream Histogram + Intermediate Fusion',])


ax = df.plot(xticks=learning_subtasks, title='Top 1 Accuracy on '+str(len(learning_subtasks))+' tasks CIL', fontsize=10)
ax.set(xlabel='Num classes', ylabel='Top-1-Accuracy')

name_fig = 'results_plot_stream_compare_XY_'+str(len(learning_subtasks))+'.png'
ax.figure.savefig(name_fig)


'''
    This script is specially designed to draw accuracy in incremental learning
'''

import pandas as pd
import pdb
import numpy as np
import re
import matplotlib.pyplot as plt

# Specify parameter here
num_tasks = 15
type_file = '.pdf'
choose_fontsize = 15

# # Numtask=5

if num_tasks == 5:

    task_id_list = [1, 2, 3, 4, 5]

    eeil_base = '0.903904 0.685297	0.596600	0.492701	0.513097'
    eeil_ours = '0.966967	0.738704	0.669271	0.578088	0.587087'

    bic_base = '0.915916	0.437826	0.501930	0.400331	0.435549'
    bic_ours = '0.912913	0.631841	0.583539	0.558001	0.590285'

    lucir_base = '0.960961	0.790915	0.660806	0.544358	0.524705'
    lucir_ours = '0.960961	0.852818	0.751241	0.649094	0.628552'

# Numtask=10

if num_tasks == 10:
    task_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    eeil_base = '0.921687	0.741018	0.689034	0.663260	0.547104	0.551141	0.535278  0.525003	0.557223	0.510065'
    eeil_ours = '0.939759	0.737988	0.693733	0.651845	0.644829	0.592621	0.578436	0.531071	0.570135	0.544270'

    bic_base = '0.795181	0.657294	0.714632	0.450735	0.573852	0.514542	0.520735	0.475074	0.498951	0.373773'
    bic_ours = '0.939759	0.759523	0.741850	0.678476	0.655118	0.603064	0.505749	0.506837	0.510810	0.458158'

    lucir_base = '0.951807	0.771229	0.730963	0.630205	0.565058	0.547013	0.533339	0.512806	0.521435	0.526432'
    lucir_ours = '0.969880	0.810295	0.739725	0.710988	0.663658	0.649562	0.642676	0.597812	0.616209	0.593135'


# Numtask=15

if num_tasks==15:
    task_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    eeil_base = '0.895652	0.597349	0.617863	0.603369	0.545172	0.558347	0.519406	0.583522	0.583927	0.542398	0.478681	0.491963	0.545068	0.514810	0.534370'
    eeil_ours = '0.947826	0.683174	0.584025	0.690562	0.624635	0.636629	0.582522	0.591433	0.602199	0.586378	0.531061	0.528817	0.514974	0.545311	0.490469'

    bic_base = '0.843478	0.726829	0.687652	0.618853	0.534009	0.516200	0.558748	0.529601	0.548862	0.511414	0.424256	0.403929	0.399130	0.389718	0.372621'
    bic_ours = '0.860870	0.750937	0.701689	0.712585	0.524174	0.506725	0.563684	0.553449	0.548573	0.400374	0.428521	0.411478	0.441220	0.463274	0.356774'

    lucir_base = '0.913043 0.696783 0.660354	0.603852	0.535672	0.563399	0.532028	0.509627	0.508601	0.494068	0.482796	0.454098	0.468355	0.440540	0.459956'
    lucir_ours = '0.930435 0.796218	0.745612	0.728013	0.673793	0.682596	0.612823	 0.626299	0.627438	0.603554	0.560842	0.519692	0.557743	0.529067	0.552131'

#######Transform to list of number###########

eeil_base = list(map(float, re.split(r'\s+', eeil_base)))
eeil_ours = list(map(float, re.split(r'\s+', eeil_ours)))

bic_base = list(map(float, re.split(r'\s+', bic_base)))
bic_ours = list(map(float, re.split(r'\s+', bic_ours)))

lucir_base = list(map(float, re.split(r'\s+', lucir_base)))
lucir_ours = list(map(float, re.split(r'\s+', lucir_ours)))


######## Multiply ########
eeil_base = list(np.array(eeil_base)*100)
eeil_ours = list(np.array(eeil_ours)*100)

bic_base = list(np.array(bic_base)*100)
bic_ours = list(np.array(bic_ours)*100)

lucir_base = list(np.array(lucir_base)*100)
lucir_ours = list(np.array(lucir_ours)*100)

plt.plot(task_id_list, eeil_base,   label = 'EEIL-Base', marker='o', color='r', linestyle='--', linewidth=2)
plt.plot(task_id_list, eeil_ours,   label = 'EEIL-IMIF (Ours)',  marker='o', color='r', linewidth=2)
plt.plot(task_id_list, bic_base, label = 'BiC-Base', marker='x', color='green', linestyle='--', linewidth=2)
plt.plot(task_id_list, bic_ours, label = 'BiC-IMIF (Ours)', marker='x', color='green', linewidth=2)
plt.plot(task_id_list, lucir_base, label = 'LUCIR-Base', marker='v', color='orange', linestyle='--', linewidth=2)
plt.plot(task_id_list, lucir_ours, label = 'LUCIR-IMIF (Ours)', marker='v', color='orange', linewidth=2)

plt.xlabel('TaskID', fontsize=choose_fontsize)
plt.ylabel('Average Accuracy', fontsize=choose_fontsize)

# remove legend
plt.legends = []

# plt.legend().set_visible(False)

# This is used to separate legend in another file
# legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=6, fontsize=25)
# def export_legend(legend, filename="legend.pdf"):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)
# export_legend(legend)


plt.xticks(task_id_list, fontsize=choose_fontsize)
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=choose_fontsize)
plt.title(str(len(task_id_list))+' tasks', fontsize=choose_fontsize)
plt.savefig('paper_average_acc_'+str(len(task_id_list))+'_tasks'+type_file)


#!/bin/bash

echo "Running eeil on Pill Base dataset"


# for NUMTASKS in 5 10 15 20

for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_early_fusion.py --approach eeil_multistream_histo \
                                    --datasets pill_base_x_multistream_true_norm\
                                    --exp-name early_fusion \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_eeil_resnet50 \
                                    --seed 4 \
                                    --gpu 3
done
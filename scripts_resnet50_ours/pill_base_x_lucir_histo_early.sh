#!/bin/bash

echo "Running lucir on Pill Base dataset"


# for NUMTASKS in 5 10 15 20

for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_early_fusion.py --approach lucir_multistream_histo \
                                    --datasets pill_base_x_multistream_true_norm\
                                    --exp-name early_fusion \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_lucir \
                                    --seed 4 \
                                    --gpu 2
done
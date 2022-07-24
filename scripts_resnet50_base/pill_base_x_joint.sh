#!/bin/bash

echo "Running on Pill Base dataset"


for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach joint \
                                    --datasets pill_base_x_true_norm \
                                    --exp-name refactor_4_exemplar \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_joint \
                                    --seed 4 \
                                    --gpu 2

done
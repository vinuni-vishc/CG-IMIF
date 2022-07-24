#!/bin/bash

echo "Running lucir on Pill Base dataset"

for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach lucir \
                                    --datasets pill_base_x_true_norm \
                                    --exp-name herding_origin \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_lucir \
                                    --exemplar-selection herding \
                                    --seed 4 \
                                    --gpu 0

done
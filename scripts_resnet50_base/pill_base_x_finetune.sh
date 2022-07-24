#!/bin/bash

echo "Running on Pill Base dataset"


for NUMTASKS in 5 10 15
# for NUMTASKS in 20
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach finetuning \
                                    --datasets pill_base_x_true_norm \
                                    --exp-name refactor_4_exemplar \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_finetune \
                                    --seed 4 \
                                    --gpu 1


done
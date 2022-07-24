#!/bin/bash

echo "Running eeil on Pill Base dataset"


# for NUMTASKS in 5 10 15 20
# for NUMTASKS in 5 10 15
# do
#     echo "NUMTASKS: $NUMTASKS"
#     python src/main_incremental.py --approach eeil \
#                                     --datasets pill_base_x \
#                                     --exp-name official \
#                                     --num-exemplars-per-class 3 \
#                                     --network resnet50 \
#                                     --num-tasks $NUMTASKS

# done

for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach bic \
                                    --datasets pill_base_x_true_norm \
                                    --exp-name refactor_4_exemplar \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../results_bic \
                                    --seed 5

done
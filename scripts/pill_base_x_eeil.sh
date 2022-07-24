#!/bin/bash

echo "Running eeil on Pill Base dataset"

# for NUMTASKS in 5 10 15
# for NUMTASKS in 15
# do
#     echo "NUMTASKS: $NUMTASKS"
#     python src/main_incremental.py --approach eeil \
#                                     --datasets pill_base_x \
#                                     --exp-name official_15_exemplar \
#                                     --num-exemplars-per-class 15 \
#                                     --network resnet50 \
#                                     --num-tasks $NUMTASKS

# done

### !!!!!!!!!!!!!!!!!!!!!! THIS IS JUST FOR TESTING !!!!!!!!!!!!!!!!!!!!!!!!!!!

for NUMTASKS in 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach eeil \
                                    --datasets pill_base_x \
                                    --exp-name recent \
                                    --num-exemplars-per-class 15 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS

done
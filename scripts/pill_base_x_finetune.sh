#!/bin/bash

echo "Running joint on Pill Base X dataset"


for NUMTASKS in 5 10 15
# for NUMTASKS in 20
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental.py --approach finetuning \
                                    --datasets pill_base_x \
                                    --exp-name official \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS

done
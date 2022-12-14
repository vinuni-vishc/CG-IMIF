#!/bin/bash

echo "Running eeil on Pill Base dataset"


# for NUMTASKS in 5 10 15 20

for NUMTASKS in 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_3stream_intermediate_fusion.py --approach eeil_3stream \
                                    --datasets pill_base_x_multistream_true_norm\
                                    --exp-name 3stream_if_recent \
                                    --num-exemplars-per-class 15 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS
done
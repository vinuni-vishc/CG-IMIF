#!/bin/bash

echo "Running eeil on Pill Large dataset"


# for NUMTASKS in 5 10 15 20
for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_multistream_contour_early.py --approach eeil_multistream_contour \
                                    --datasets pill_base_x_multistream \
                                    --exp-name official \
                                    --num-exemplars-per-class 15 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS

done
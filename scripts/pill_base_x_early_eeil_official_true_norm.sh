#!/bin/bash

echo "Running eeil on Pill Large dataset"


# for NUMTASKS in 5 10 15 20
for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_early_fusion.py --approach eeil_multistream_histo \
                                    --datasets pill_base_x_multistream_true_norm \
                                    --exp-name early_refactor_4_exemplar \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --seed 3
done
# CG-IMIF: Multi-stream Fusion for Class Incremental Learning in Pill Image Classification

## Introduction

Classifying pill categories from real-world images is crucial for various smart healthcare applications. Although existing approaches
in image classification might achieve a good performance on fixed pill
categories, they fail to handle novel instances of pill categories that are
frequently presented to the learning algorithm. To this end, a trivial solution is to train the model with novel classes. However, this may result
in a phenomenon known as catastrophic forgetting, in which the system
forgets what it learned in previous classes. In this paper, we address
this challenge by introducing the class incremental learning (CIL) ability to traditional pill image classification systems. Specifically, we propose a novel incremental multi-stream intermediate fusion framework enabling incorporation of an additional guidance information stream that
best matches the domain of the problem into various state-of-the-art
CIL methods. From this framework, we consider color-specific information of pill images as a guidance stream and devise an approach,
namely “Color Guidance with Multi-stream intermediate fusion”(CGIMIF) for solving CIL pill image classification task. We conduct comprehensive experiments on real-world incremental pill image classification
dataset, namely VAIPE-PCIL, and find that the CG-IMIF consistently
outperforms several state-of-the-art methods by a large margin in different task settings.

![General Pipeline](/figures/pipeline_overview.png)

## Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)
- [PyTorch >= 1.7](https://pytorch.org/)

### Installation

```bash
conda env create CG_IMIF -f environment.yml
conda activate CG_IMIF
```

## Dataset Preparation
### VAIPE: A Large-scale and Real-World Open Pill Image Dataset for Visual-based Medicine Inspection
This dataset was developed by our [VAIPE team](https://vaipe.org/) to facilitate AI research in pill and prescription images. In this research, we derive a different data version, namely VAIPE-PCIL (VAIPE Pill Class Incremental Learning) for empiricial study of incremental learning behaviour on pill image classification problem. We have already prepared and processed VAIPE-PCIL, and it can be accessed [here](link_to_dataset)

## Decomposition of Multi-stream Pill CIL
We define a multi-stream class incremental learning model M as a combination
of three key components: 
1) Single stream base method X, 
2) Additional stream of information Y
3) Method of fusing stream Z.

```
M = Base method X + Feature stream Y + Fusion mechanism Z
```

### Base method X
We aim to exploit the capability of exemplar-based CIL methods by attaching these to our proposed
framework. Therefore, we chose a few representative methods such as End-to-End Incremental Learning, BiC, 

### Feature stream Y
Edge, and color histogram are two main features that we aim to investigate the effect in a pill incremental learning system. For further infomation about how to obtain
these features, we recommend to take a look at ```src/datasets/base_dataset.py``` and ```benchmarked_pillCIL_data/create_edge_data_folder.py```

### Fusion mechanism Z
Two fusion techniques: early fusion, and intermediate fusion are investigated and combined with previous feature stream Y to compare the effectiveness of tackling catastrophic forgetting effect. All of related scripts for running early and intermediate fusion can be found in ```scripts_resnet50_base``` and ```scripts_resnet50_ours```

From the aforementioned decomposition, our CG-IMIF replaces:1) the representative stream Y with color-specific information, and 2) the fusion
technique Z with the proposed IMIF

![CG-IMIF](/figures/X-CG-IMIF.png)

Figure 2: Our proposed CG-IMIF architecture composes of: 1) color histogram
feature extraction (orange block), and 2) intermediate fusion framework (purple
block) to incorporate additional information stream.

## Training and Evaluation
These two phases can be executed via our provided bash file in ```scripts_resnet50_base``` and ```scripts_resnet50_ours```. An example of executing training
and evaluation script with: LUCIR base method, color histogram stream, and intermediate fusion can be:

```
#!/bin/bash

echo "Running lucir on Pill Base dataset"


for NUMTASKS in 5 10 15
do
    echo "NUMTASKS: $NUMTASKS"
    python src/main_incremental_intermediate_fusion.py --approach lucir_multistream_histo \
                                    --datasets pill_base_x_multistream_true_norm \
                                    --exp-name refactor_4_exemplar \
                                    --num-exemplars-per-class 4 \
                                    --network resnet50 \
                                    --num-tasks $NUMTASKS \
                                    --results-path ../new_results_lucir \
                                    --seed 4 \
                                    --gpu 2

done
```

## Results
We evaluate our proposed CG-IMIF approach and report the overall performance in comparison with several state-of-the-art approaches. Experimental results show that most of the state-of-the-art approaches attached with our proposed IMIF tool and color-specific information as additional stream help
to achieve consistent improvements over task settings. The setting consists of
three tasks in total where the number of categories is uniformly distributed
for 5, 10, and 15 tasks.

![accuracy](/figures/results1.png)

Figure 3: Incremental accuracy for different task settings among the original
version and our method CG-IMIF.

![ForgettingRate](/figures/results2.png)

Figure 4: Incremental accuracy for different task settings among the original
version and our method CG-IMIF.


## Acknowledgement
Our CG-IMIF implementation is inspried from [FACIL library](https://github.com/mmasana/FACIL). Also, this research would not fulfilled without support of VAIPE team and VinUni-Illinois Smart Health Center.

## Contact
If you have any concerns or support need on this repository, please drop me an email at ```nguyentrongtung11101999@gmail.com```

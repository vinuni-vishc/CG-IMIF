# CG-IMIF: Multi-stream Fusion for Class Incremental Learning in Pill Image Classification


An official implementation for my research work "Multi-stream Class Incremental Learning for Pill Image Classification"  at VinUni-Illinois  Smart Health Center

## Introduction

In this work, we address the common pill image classification problem with a novel learning capability: Class incremental learning (CIL). Specifically, we 
propose a novel incremental multi-stream intermediate fusion framework enabling incorporation of an additional guidance information stream that best matches the domain of the problem into various state-of-the-art CIL
methods. From this framework, we consider color-specific information of pill images as a guidance stream and devise an approach, namely “Color Guidance with Multi-stream intermediate fusion”(CG-IMIF) for solving
CIL pill image classification task.

## Usage 

The implementation is employed from available CIL framework []. We inherit the base implementation and modify various modules for our proposed method. For setting up
environments, please refer to the original instructions from []

### Dataset
Most of current pill image classification dataset lacks of the authentic images describing the practical usage of pill in real-world scenario. In our project,
we released a novel dataset namely VAIPE-Pill. The formal description of the VAIPE-Pill dataset and further information about other dataset can be found at ...

## Training and Evaluation
Training and testing are conducted for the base method X with and without our proposed framework CG-IMIF. The experiments are performed with similar settings for both
version for fair comparison. 

The running scripts for training and evaluation for base method without our proposed framework CG-IMIF can be found at ```scripts_resnet50_base```. On the other hand,
we provided the scripts for the remaining version which integrate CG-IMIF framework with base method at ```scripts_resnet50```. Note that it is compulsory to iterate the
seed parameters with similar values across methods for fair comparision.

## Results
We conduct comprehensive experiments on real-world incremental pill image classification dataset, namely VAIPE-PCIL, and find that the CG-IMIF consistently outperforms several state-of-the-art methods by a large margin. More importantly, our
method demonstrates its efficacy in terms of both average accuracy and average forgetting rate across a variety of task settings.

## Acknowledgement
This work would not be fully achieved without constructive comments and arguments from my team VAIPE.

##

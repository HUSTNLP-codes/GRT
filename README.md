# Graph Reasoning Transformers for Knowledge-Aware Question Answering
Augmenting Language Models (LMs) with structured knowledge graphs (KGs) aims to leverage structured world knowledge to enhance the capability of LMs to complete knowledge-intensive tasks. However, existing methods are unable to effectively utilize the structured knowledge in a KG due to their inability to capture the rich relational semantics of knowledge triplets. Moreover, the modality gap between natural language text and KGs has become a challenging obstacle when aligning and fusing cross-modal information. To address these challenges, we propose a novel knowledge-augmented question answering (QA) model, namely, Graph Reasoning Transformers (GRT). Different from conventional node-level methods, the GRT serves knowledge triplets as atomic knowledge and utilize a triplet-level graph encoder to capture triplet-level graph features. Furthermore, to alleviate the negative effect of the modality gap on joint reasoning, we propose a representation alignment  pretraining to align the cross-modal representations and introduce a cross-modal information fusion module with attention bias to enable cross-modal information fusion. 

## Setup

- Setup conda environment
```
conda create -n GRT python=3.8
conda activate GRT
```
- Install packages with a setup file
```
bash setup.sh
```
- Download data

You can download all the preprocessed data with the [link](https://drive.google.com/file/d/1jEe_T4ZNM4rk0FMDcL8E-sRI2eC5O_qj/view?usp=drive_link).


## Pre-train
```
bash pre-train.sh
```

## Fine-tune
```
bash fine-tune.sh
```

## Acknowledgement
This repo is built upon the QAT:
```
https://github.com/mlvlab/QAT
```

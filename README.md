# RumorMixer: Exploring Echo Chamber Effect and Platform Heterogeneity for Rumor Detection

## Overview

This is the code for our paper **RumorMixer: Exploring Echo Chamber Effect and Platform Heterogeneity for Rumor Detection**.
It is a differentiable architecture search framework based on GraphMixer for rumor detection.

## Requirements:

To execute this project, it's essential to install the required dependencies. To do so, navigate to the directory containing the `requirements.txt` file and execute the following command:

```
pip install -r requirements.txt
```

## Instructions to run the experiment

To execute the project, run the following command in your terminal:

**Step 1.** Run the search process, given different random seeds.
(The _Twitter16_ dataset is used as an example)

```bash
python Libs/Exp/search.py
```

The results are saved in the directory `Output/Logs/`, e.g., `Search-RumorMixer-Twitter`.

**Step 2.** Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.

```bash
python Libs/Exp/finetune.py
```

Step 2 is a coarse-graind tuning process, and the results are saved in a picklefile in the directory `Finetune`.

## Citing Our Work

If you find this project useful and use it in your research, please consider citing our paper.

## Acknowledgement

The code is built on [DARTS](https://github.com/quark0/darts), one of the most well-known differentiable architecture search methods and [SANE](https://github.com/LARS-research/SANE), a differentiable architecture search for graph neural network (GNN).

## Misc

If you have any questions about this project, you can open issues, thus it can help more people who are interested in this project. We will reply to your issues as soon as possible.
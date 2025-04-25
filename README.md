# FlowerTune LLM on Medical Dataset

### Introduction

This directory conducts federated instruction tuning with a pretrained [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

### Evaluation in the three baseline datasets with the proposed approach:

|         | PubMedQA | MedMCQA | MedQA | CareQA |  Avg  |
| :-----: | :------: | :-----: | :---: | :---:  | :---: |
| Acc (%) |   66.20  |  60.29  | 68.42 | 53.64  | 62.14 |

#### Communication budget: 1040.31 MB*
*Note that this value has been obtained when running the experiment using a NVIDIA GPU Tesla V100-PCIE-32GB.

## Changes from baseline

* Following the advances obtained with the approach presented by the [Gachon Cognitive Computing Lab](https://github.com/gachon-CCLab/GCCL-Medical-LLM-FlowerTune), we have used as a base model the [ContactDoctor/Bio-Medical-Llama-3-8B](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) fine tuned model.
* We train the model during 5 rounds, `num-server-rounds = 5`, see [peft_5](https://github.com/judithspd/ai4os-fedllm-medical-v2/tree/main/flowertune-eval-medical/peft_5).
* We train the model locally during 3 epochs: `train.training-arguments.num-train-epochs = 3`.
* We take `train.learning-rate-max = 5e-6` and `train.learning-rate-min = 1e-7`.
* We use [FedAvgOpt](https://arxiv.org/abs/2501.15949) as aggregation function.

## Methodology

This baseline performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.1) of the total nodes to participate in each round, for a total of `5` rounds.
All settings are defined in `pyproject.toml`.


## Running the experiment

First, login in huggingface:
```bash
huggingface-cli login
```

Then, run the experiment:

```bash
flwr run .
```

Evaluation in the three baseline datasets:

```bash
python eval.py --base-model-name-path="ContactDoctor/Bio-Medical-Llama-3-8B" --peft-path="peft_5" --batch-size=16 --quantization=4 --datasets=pubmedqa,medmcqa,medqa,careqa
```



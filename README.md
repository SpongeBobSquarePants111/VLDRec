# VLDRec

This repository is the official implementation of **Alleviating Video Length Effect of Micro-video Recommendation** 

## Requirements

We conducted experiments under：

- python 3.7
- pytorch1.5.1

The experiments are conducted on a single Linux server with AMD Ryzen Threadripper 2990WX@3.0GHz, 128G RAM and 4 NVIDIA GeForce RTX 2080TI-11GB.

## Data

In `./data`  folder, `./train.pkl`, `./valid.pkl` and `./test.pkl` are data generated from the open source `<u>`kuaishou`</u>` dataset, and other files are auxiliary files generated from these three files for the convenience of training

## Training

```shell
$ ./run.sh
```

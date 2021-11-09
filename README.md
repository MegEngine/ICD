# Instance-conditional Knowledge Distillation for Object Detection
This is a [MegEngine](https://github.com/MegEngine/MegEngine) implementation of the paper "Instance-conditional Knowledge Distillation for Object Detection", based on [MegEngine Models](https://github.com/MegEngine/Models).

The pytorch implementation based on detectron2 will be released soon.

> [**Instance-Conditional Knowledge Distillation for Object Detection**](https://arxiv.org/abs/2110.12724),            
> Zijian Kang, Peizhen Zhang, Xiangyu Zhang, Jian Sun, Nanning Zheng         
> In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2021            
> [[arXiv](https://arxiv.org/abs/2106.14855)]

## Requirements

### Installation

In order to run the code, please prepare a CUDA environment with:
- Python 3 (3.6 is recommended)
- [MegEngine](https://github.com/MegEngine/MegEngine)


1. Install dependancies.

```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

2. Prepare [MS-COCO 2017 dataset](http://cocodataset.org/#download)，put it to a proper directory with the following structures:

```
/path/to/
    |->coco
    |    |annotations
    |    |train2017
    |    |val2017
```


[Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. European Conference on Computer Vision (ECCV), 2014.

## Usage

### Train baseline models

Following [MegEngine Models](https://github.com/MegEngine/Models):
```bash
python3 train.py -f distill_configs/retinanet_res50_coco_1x_800size.py -n 8 \
                       -d /data/Datasets
```

`train.py` arguments：

- `-f`, config file for the network.
- `-n`, required devices(gpu).
- `-w`, pretrained backbone weights.
- `-b`, training `batch size`, default is 2.
- `-d`, dataset root，default is `/data/datasets`.


### Train with distillation

```bash
python3 train_distill_icd.py -f distill_configs/retinanet_res50_coco_1x_800size.py \ 
    -n 8 -l -d /data/Datasets -tf configs/retinanet_res101_coco_3x_800size.py \
    -df distill_configs/ICD.py \
    -tw _model_zoo/retinanet_res101_coco_3x_800size_41dot4_73b01887.pkl
```

`train_distill_icd.py` arguments：

- `-f`, config file for the student network.
- `-w`, pretrained backbone weights.
- `-tf`, config file for the teacher network.
- `-tw`, pretrained weights for the teacher.
- `-df`, config file for the distillation module, `distill_configs/ICD.py` by default. 
- `-l`, use the inheriting strategy, load pretrained parameters.
- `-n`, required devices(gpu).
- `-b`, training `batch size`, default is 2.
- `-d`, dataset root，default is `/data/datasets`.

Note that we set `backbone_pretrained` in distill configs, where backbone weights will be loaded automatically, that `-w` can be omitted. Checkpoints will be saved to a log-xxx directory.

### Evaluate

```
python3 test.py -f distill_configs/retinanet_res50_coco_3x_800size.py -n 8 \
     -w log-of-xxx/epoch_17.pkl -d /data/Datasets/
```

`test.py` arguments：

- `-f`, config file for the network.
- `-n`, required devices(gpu).
- `-w`, pretrained weights.
- `-d`, dataset root，default is `/data/datasets`.

## Examples and Results
### Steps
1. Download the pretrained teacher model to ```_model_zoo``` directory.
2. Train baseline or distill with ICD.
3. Evaluate checkpoints (use the last checkpoint by default).

### Example of Common Detectors

#### RetinaNet
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár. IEEE International Conference on Computer Vision (ICCV), 2017.


- Teacher RetinaNet-R101-3x:
https://data.megengine.org.cn/models/weights/retinanet_res101_coco_3x_800size_41dot4_73b01887.pkl


- Config: distill_configs/retinanet_res50_coco_1x_800size.py

Command: 
```
python3 train_distill_icd.py -f distill_configs/retinanet_res50_coco_1x_800size.py \
    -n 8 -l -d /data/Datasets -tf configs/retinanet_res101_coco_3x_800size.py \
    -df distill_configs/ICD.py \
    -tw _model_zoo/retinanet_res101_coco_3x_800size_41dot4_73b01887.pkl
```

#### FCOS

- [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355) Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. IEEE International Conference on Computer Vision (ICCV), 2019.

- Teacher FCOS-R101-3x:
https://data.megengine.org.cn/models/weights/fcos_res101_coco_3x_800size_44dot3_f38e8df1.pkl


- Config: distill_configs/fcos_res50_coco_1x_800size.py

Command: 
```
python3 train_distill_icd.py -f distill_configs/fcos_res50_coco_1x_800size.py \
    -n 8 -l -d /data/Datasets -tf configs/fcos_res101_coco_3x_800size.py \
    -df distill_configs/ICD.py \
    -tw _model_zoo/fcos_res101_coco_3x_800size_44dot3_f38e8df1.pkl
```

#### ATSS

- [Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection](https://arxiv.org/abs/1912.02424) Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z. Li. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

- Teacher ATSS-R101-3x:
https://data.megengine.org.cn/models/weights/atss_res101_coco_3x_800size_44dot7_9181687e.pkl


- Config: distill_configs/atss_res50_coco_1x_800size.py

Command: 
```
python3 train_distill_icd.py -f distill_configs/atss_res50_coco_1x_800size.py \
    -n 8 -l -d /data/Datasets -tf configs/atss_res101_coco_3x_800size.py \
    -df distill_configs/ICD.py \
    -tw _model_zoo/atss_res101_coco_3x_800size_44dot7_9181687e.pkl
```

### Results of AP in MS-COCO:

| Model         | Baseline     | +ICD          | 
| ---           | :---:        | :---:         |
| Retinanet     | 36.8         | 40.3          |
| FCOS          | 40.0         | 43.3          |
| ATSS          | 39.6         | 43.0          |


### Notice

- Results of this implementation are mainly for demonstration, please refer to the Detectron2 version for reproduction. 

- We simply adopt the hyperparameter from Detectron2 version, further tunning could be helpful.

- There is a known CUDA memory issue related to MegEngine: the actual memory consumption will be much larger than the theoretical value, due to the memory fragmentation. This is expected to be fixed in a future version of MegEngine.

## Acknowledgement

This repo is modified from [MegEngine Models](https://github.com/MegEngine/Models). We also refer to [Pytorch](https://github.com/pytorch/pytorch), [DETR](https://github.com/facebookresearch/detr) and [Detectron2](https://github.com/facebookresearch/detectron2) for some implementations.

## License

This repo is licensed under the Apache License, Version 2.0 (the "License").

## Citation
```
@inproceedings{kang2021icd,
    title={Instance-conditional Distillation for Object Detection},
    author={Zijian Kang, Peizhen Zhang, Xiangyu Zhang, Jian Sun, Nanning Zheng},
    year={2021},
    booktitle={NeurIPS},
}
```
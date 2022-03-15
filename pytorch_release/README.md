# Instance-Conditional Knowledge Distillation for Object Detection
This is an official implementation of the paper "Instance-Conditional Knowledge Distillation for Object Detection" in [Pytorch](https://pytorch.org), it supports various detectors from Detectron2 and AdelaiDet. 


# Requirements
The project is depending on the following libraries. You may need to install Detectron2 and AdelaiDet mannully, please refer to their github pages.
- Python3 (recommand 3.8)
- pytorch == 1.9.0
- torchvision == 0.10.0
- opencv-python == 4.5.4.58
- [Detectron2](https://github.com/facebookresearch/detectron2) == 0.5.0
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) == 7bf9d87 

(To avoid conflict, we recommend to use the exact above versions.)

Reference command for installation:
```
# Switch to this directory
pip install pip --upgrade
pip install -r requirements.txt
pip install https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.5.tar.gz
pip install 'git+https://github.com/aim-uofa/AdelaiDet.git@7bf9d87'
```

You will also need to prepare datasets according to [detectron2](https://github.com/facebookresearch/detectron2/tree/main/datasets), put your data under the following structure, and set the environment variable by `export DETECTRON2_DATASETS=/path/to/datasets`.
```
$DETECTRON2_DATASETS/
  coco/
  annotations/
    instances_{train,val}2017.json
    {train,val}2017/
      # image files
```

# Usage
## Train baseline models
We use [train_baseline.py](./train_baseline.py) to train baseline models, it is very similar to [tools/train_net.py](https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py). 

You can use any config files for detectron2 or AdelaiDet to specify a training setting.
```
usage: train_baseline.py [-h] [--config-file FILE] [--resume] [--eval-only]
                         [--num-gpus NUM_GPUS] [--num-machines NUM_MACHINES]
                         [--machine-rank MACHINE_RANK] [--dist-url DIST_URL]
                         ...

positional arguments:
  opts                  Modify config options at the end of the command. For
                        Yacs configs, use space-separated "PATH.KEY VALUE"
                        pairs. For python-based LazyConfig, use
                        "path.key=value".

optional arguments:
  -h, --help            show this help message and exit
  --config-file FILE    path to config file
  --resume              Whether to attempt to resume from the checkpoint
                        directory. See documentation of
                        `DefaultTrainer.resume_or_load()` for what it means.
  --eval-only           perform evaluation only
  --num-gpus NUM_GPUS   number of gpus *per machine*
  --num-machines NUM_MACHINES
                        total number of machines
  --machine-rank MACHINE_RANK
                        the rank of this machine (unique per machine)
  --dist-url DIST_URL   initialization URL for pytorch distributed backend.
                        See https://pytorch.org/docs/stable/distributed.html
                        for details.
```
### Examples:

Train a retinanet baseline detector on a single machine:

```
train_baseline.py --num-gpus 8 --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml
```

Change some config options:

```
train_baseline.py --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
```

Run on multiple machines:
```
(machine0)$ train_baseline.py --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
(machine1)$ train_baseline.py --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
```

## Train and distill models
We leave everything the same as the above, except the entry ([train_distill.py](./train_distill.py)) and the config. 

### Examples:

Train RetinaNet with distillation:

```
python3 train_distill.py --num-gpus 8 --resume --config-file configs/Distillation-ICD/retinanet_R_50_R101_icd_FPN_1x.yaml OUTPUT_DIR output/icd_retinanet
```

Train Faster R-CNN with distillation:

```
python3 train_distill.py --num-gpus 8 --resume --config-file configs/Distillation-ICD/RCNN_R_50_R101_icd_FPN_1x.yaml OUTPUT_DIR output/icd_frcnn
```

Train CondInst with distillation:

```
python3 train_distill.py --num-gpus 8 --resume --config-file configs/Distillation-ICD/CondInst_R50_R101_icd.yaml OUTPUT_DIR output/icd_condinst
```

### Write distillation configs:
To introduce how to write a config for distillation, let's see two examples:

**If teacher model is released by detectron2 officially:**

You can load checkpoint from detectron2 model_zoo API, set `MODEL_LOAD_OFFICIAL=True` and use the corresponding config file. You may also set `WEIGHT_VALUE` to the desired number. 

```
MODEL:
  DISTILLER:
    MODEL_LOAD_OFFICIAL: True
    MODEL_DISTILLER_CONFIG: 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
      
    INS_ATT_MIMIC:
      WEIGHT_VALUE: 8.0
```

Note: It also support configs from detectron2 new baselines, like [LSJ (large scale jitters) models](https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py), which could be helpful in practice.


**If you want to use a standalone teacher trained by yourself:**

If you train a teacher by ourselves, you may need to define a standalone config for the teacher. Set `MODEL_LOAD_OFFICIAL=False` and use a standalone config file.

``` 
MODEL:
  DISTILLER:
    MODEL_LOAD_OFFICIAL: False
    MODEL_DISTILLER_CONFIG: 'Teachers/SOLOv2_R101_3x_ms.yaml'
      
    INS_ATT_MIMIC:
      WEIGHT_VALUE: 8.0
```

For teacher's config, simply set pretrained weight to a checkpoint file:
```
_BASE_: "../Base-SOLOv2.yaml"
MODEL:
  WEIGHTS: "https://cloudstor.aarnet.edu.au/plus/s/9w7b3sjaXvqYQEQ" 
  # This is the official release from AdelaiDet.
  RESNETS:
    DEPTH: 101
```

You can find more options in [utils/build.py](utils/build.py)

# Results
For object detection in MS-COCO:
| Model         | Baseline (BoxAP)     | + Ours (BoxAP)           | 
| ---           | :---:        | :---:         |
| Faster R-CNN     | 37.9         | 40.9 (+3.0)        |
| Retinanet     | 37.4         | 40.7 (+3.3)         |
| FCOS          | 39.4         | 42.9 (+3.5)         |

For instance-segmentation in MS-COCO:
| Model         | Baseline (BoxAP)    | + Ours (BoxAP)          | Baseline (MaskAP)    | + Ours (MaskAP)          | 
| ---           | :---:        | :---:         | :---:        | :---:         |
| Mask R-CNN     | 38.6        | 41.2 (+2.6)         |  35.2 | 37.4 (+2.2) |
| SOLOv2     | - | - | 34.6 | 38.5 (+3.9) |
| CondInst        |39.7 | 43.7 (+4.0) | 35.7 | 39.1 (+3.4) |
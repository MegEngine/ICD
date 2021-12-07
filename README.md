# Instance-Conditional Knowledge Distillation for Object Detection
This is the official implementation of the paper "Instance-Conditional Knowledge Distillation for Object Detection", based on [MegEngine](./megengine_release/README.md) and [Pytorch](./pytorch_release/README.md). Go to the desired subfolders for more information and guidance!


<div align="center">
  <img src="Poster.png"/>
</div>

> [**Instance-Conditional Knowledge Distillation for Object Detection**](https://arxiv.org/abs/2110.12724),            
> Zijian Kang, Peizhen Zhang, Xiangyu Zhang, Jian Sun, Nanning Zheng         
> In Proc. of Advances in Neural Information Processing Systems (NeurIPS), 2021            
> [[arXiv](https://arxiv.org/abs/2110.12724)][[Citation](#citation)][[OpenReview](https://openreview.net/forum?id=k7aeAz4Vbb)]

## Usage 
You can find two implementation for [MegEngine](./megengine_release/README.md) and [Pytorch](./pytorch_release/README.md) in two sub-folders. We use the latter one to report performance in the paper. Switch to the subfolder for more information.

### Try it in a few lines :
Take the detectron2 implementation as an example, you can train your model in a few lines:
```
cd pytorch_release

# Installation
pip install pip --upgrade
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/aim-uofa/AdelaiDet.git'

# Prepare dataset according to https://github.com/facebookresearch/detectron2/tree/main/datasets

# train and distill a retinanet detector with ICD
python3 train_distill.py --num-gpus 8 --resume --config-file configs/Distillation-ICD/retinanet_R_50_R101_icd_FPN_1x.yaml OUTPUT_DIR output/icd_retinanet
```

## Performance
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

## Acknowledgement

Some files are modified from [MegEngine Models](https://github.com/MegEngine/Models) and [Detectron2](https://github.com/facebookresearch/detectron2). We also refer to [Pytorch](https://github.com/pytorch/pytorch), [DETR](https://github.com/facebookresearch/detr) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet) for some implementations. 


## License

This repo is licensed under the Apache License, Version 2.0 (the "License").

## Citation
You can use the following BibTeX entry for citation in your research.
```
@inproceedings{kang2021instance,
  title={Instance-Conditional Knowledge Distillation for Object Detection},
  author={Kang, Zijian and Zhang, Peizhen and Zhang, Xiangyu and Sun, Jian and Zheng, Nanning},
  booktitle={In Proc. of the Thirty-Fifth Conference on Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

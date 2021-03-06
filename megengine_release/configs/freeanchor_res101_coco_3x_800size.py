# -*- coding: utf-8 -*-
# This repo is licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

import models


class CustomFreeAnchorConfig(models.FreeAnchorConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet101"


@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "freeanchor_res101_coco_3x_800size_43dot9_8c707d7d.pkl"
)
def freeanchor_res101_coco_3x_800size(**kwargs):
    r"""
    FreeAnchor trained from COCO dataset.
    `"FreeAnchor" <https://arxiv.org/abs/1909.02466>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = models.FreeAnchorConfig()
    cfg.backbone_pretrained = False
    return models.FreeAnchor(cfg, **kwargs)


Net = models.FreeAnchor
Cfg = CustomFreeAnchorConfig

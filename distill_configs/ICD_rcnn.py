import megengine.module as M
import megengine.functional as F
from models.ICD.ICD import ICD
from easydict import EasyDict as edict

def get_distillator():
    cfg = edict({
        'distiller': {
            'FEAT_KEYS': ['p2', 'p3', 'p4', 'p5', 'p6'],
            'WEIGHT_VALUE': 3.0,
            'TEMP_VALUE': 1.0,
            'HIDDEN_DIM': 256,
            'NUM_CLASSES': 80,
            'MAX_LABELS': 300,
            'ATT_HEADS': 8,
            'USE_POS_EMBEDDING': True,
            'DECODER_POSEMB_ON_V': False,

        },
    })
    return ICD(256, cfg)

Net = get_distillator
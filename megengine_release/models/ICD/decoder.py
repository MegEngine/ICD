import megengine as mge
import megengine.module as M
from megengine import functional as F
import numpy as np
from .transformer import MultiheadAttention
#from .utility import has_nan_or_inf

# mge.core.set_option('async_level', 0)

class DecoderWrapper(M.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.distiller.HIDDEN_DIM
        heads = cfg.distiller.ATT_HEADS

        # this is a local module derived from official implementation, we modify the last modules
        self.matt = MultiheadAttention(channels, heads)

        self.pos_projector = M.Linear(in_features=channels, out_features=channels)
        self.use_pos = cfg.distiller.USE_POS_EMBEDDING
        self.pos_on_v = cfg.distiller.DECODER_POSEMB_ON_V

    def with_pos_embed(self, tensor, pos):
        '''
        tensor: [S, N, C]
        pos: [S, N, C] or [S, 1, C]
        '''
        if not self.use_pos:
            return tensor

        pos = self.pos_projector(pos)
        return tensor if pos is None else tensor + pos


    def forward(self, q, k, v, query_mask=None, key_padding_mask=None, pos_embedding=None, proj_only=False):
        # q, v: [sequence_len, batch_size, channels]
        k = self.with_pos_embed(k, pos_embedding)
        if self.pos_on_v:
            v = self.with_pos_embed(v, pos_embedding)
        att, mask, values = self.matt(
            q, k, v, key_padding_mask=key_padding_mask, proj_only=proj_only)
        return att, mask, values

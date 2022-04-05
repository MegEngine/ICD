# Some codes are modified from pytorch

import megengine as mge
import megengine.module as M
from megengine import functional as F
from megengine import Parameter
from megengine.module.init import xavier_uniform_, zeros_
from typing import List, Tuple, Dict, Optional
import numpy as np
from .utility import safe_masked_fill, has_nan_or_inf


def multi_head_attention_forward(
    query: mge.Tensor,
    key: mge.Tensor,
    value: mge.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: mge.Tensor,
    in_proj_bias: Optional[mge.Tensor],
    bias_k: Optional[mge.Tensor],
    bias_v: Optional[mge.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: mge.Tensor,
    out_proj_bias: Optional[mge.Tensor],
    training: bool = True,
    key_padding_mask: Optional[mge.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[mge.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[mge.Tensor] = None,
    k_proj_weight: Optional[mge.Tensor] = None,
    v_proj_weight: Optional[mge.Tensor] = None,
    static_k: Optional[mge.Tensor] = None,
    static_v: Optional[mge.Tensor] = None,
    proj_only: bool = False
) -> Tuple[mge.Tensor, Optional[mge.Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.shape
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]

    if isinstance(embed_dim, mge.Tensor):
        # embed_dim can be a tensor when JIT tracing
        #head_dim = embed_dim.div(num_heads, rounding_mode='trunc')

        #NOTE: when positive number, floor_div is equivalent to trunc_div (in megengine only floor_div is available)
        head_dim = F.floor_div(embed_dim, num_heads)
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    assert not use_separate_proj_weight
    assert need_weights
    assert attn_mask is None
    assert bias_k is None and bias_v is None
    assert not add_zero_attn

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = 0
    _end = embed_dim
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    q = F.linear(query, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim
    _end = embed_dim * 2
    _w = in_proj_weight[_start:_end, :]
    if _b is not None:
        _b = _b[_start:_end]
    k = F.linear(key, _w, _b)

    # This is inline in_proj function with in_proj_weight and in_proj_bias
    _b = in_proj_bias
    _start = embed_dim * 2
    _end = None
    _w = in_proj_weight[_start:, :]
    if _b is not None:
        _b = _b[_start:]
    v = F.linear(value, _w, _b)

    q = q * scaling
    raw_v = v
    raw_v = raw_v.reshape(-1, bsz, num_heads, head_dim)
    if proj_only:
        return query, None, raw_v
    

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == np.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.astype(np.bool)


    #def _pad_last_dim_right_only(tensor):
    #    '''
    #    To replace with torch.nn.functional.pad(tensor, (0, 1))
    #    '''
    #    return F.concat([tensor, F.expand_dims(F.zeros(tensor.shape[:-1]), axis=-1)], axis=-1)

    #q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(1, 0, 2)

    if k is not None:
        #k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)

    if v is not None:
        #v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)

    if static_k is not None:
        assert static_k.shape[0] == bsz * num_heads
        assert static_k.shape[2] == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.shape[0] == bsz * num_heads
        assert static_v.shape[2] == head_dim
        v = static_v

    src_len = k.shape[1]

    if key_padding_mask is not None:
        assert key_padding_mask.shape[1] == src_len

    #attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    attn_output_weights = F.matmul(q, k.transpose(0, 2, 1))

    assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len)
        key_padding_mask = F.expand_dims(F.expand_dims(key_padding_mask, axis=1), axis=2)
        attn_output_weights = safe_masked_fill(attn_output_weights, key_padding_mask, float("-inf"))
        attn_output_weights = attn_output_weights.reshape(bsz * num_heads, tgt_len, src_len)

    attn_output_weights_no_softmax = attn_output_weights
    attn_output_weights = F.softmax(attn_output_weights, axis=-1)
    attn_output_weights = F.dropout(attn_output_weights, dropout_p, training=training)

    attn_output = F.matmul(attn_output_weights, v)
    assert attn_output.shape == (bsz * num_heads, tgt_len, head_dim)
    attn_output = F.transpose(attn_output, (1, 0, 2)).reshape(tgt_len, bsz, embed_dim)
    attn_output = F.nn.linear(attn_output, out_proj_weight, out_proj_bias)

    # average attention weights over heads
    attn_output_weights = attn_output_weights_no_softmax.reshape(bsz, num_heads, tgt_len, src_len)

    return attn_output, attn_output_weights, raw_v


class MultiheadAttention(M.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[mge.Tensor]
    bias_v: Optional[mge.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # True By default
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        # False By default
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(F.zeros((3 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = Parameter(F.zeros((3 * embed_dim,)))
        else:
            self.in_proj_bias = None
        self.out_proj = M.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(F.zeros((1, 1, embed_dim)))
            self.bias_v = Parameter(F.zeros((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            zeros_(self.in_proj_bias)
            zeros_(self.out_proj.bias)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: mge.Tensor, key: mge.Tensor, value: mge.Tensor, key_padding_mask: Optional[mge.Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[mge.Tensor] = None, proj_only=False) -> Tuple[mge.Tensor, Optional[mge.Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.
          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        attn_output, attn_output_weights, values = multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights, #need_weights by default is True
            attn_mask=attn_mask, proj_only=proj_only)
        return attn_output, attn_output_weights, values

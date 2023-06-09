# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
import paddle


class HOITransformerTS(paddle.nn.Layer):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm =paddle.nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        ###################################################################################
        decoder_norm =paddle.nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayerAttMap(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        self.decoder = TransformerDecoderTS(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        ##################################################################################
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                paddle.nn.initializer.XavierUniform()(p)
                # nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed1,
                query_embed_q=None, query_embed_e=None,
                query_embed2_mask=None,
                pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        #####################################################################
        query_embed1 = query_embed1.unsqueeze(1).repeat(1, bs, 1)
        tgt1 = paddle.zeros_like(query_embed1)
        if query_embed_q is not None:
            att_s, att_t, hs, hs_t = self.decoder.forwardt(tgt1, memory,
                                                           memory_key_padding_mask=mask,
                                                           tgt_key_padding_mask_t=query_embed2_mask,
                                                           pos=pos_embed,
                                                           query_pos=query_embed1,
                                                           query_embed_q=query_embed_q,
                                                           query_embed_e=query_embed_e)

            return att_s, att_t, hs.transpose(1, 2), hs_t.transpose(1, 2),

        else:
            hs = self.decoder(tgt1, memory,
                              memory_key_padding_mask=mask,
                              pos=pos_embed,
                              query_pos=query_embed1)
            return hs.transpose(1, 2)


class TransformerEncoder(paddle.nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask = None,
                src_key_padding_mask = None,
                pos= None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderTS(paddle.nn.Layer):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask= None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        output = tgt
        hs = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)[0]
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
        return paddle.stack(hs)

    def forwardt(self, tgt, memory,
                 tgt_mask= None,
                 tgt_mask_t= None,
                 memory_mask= None,
                 tgt_key_padding_mask= None,
                 tgt_key_padding_mask_t = None,
                 memory_key_padding_mask = None,
                 pos = None,
                 query_pos = None,
                 query_embed_q = None,
                 query_embed_e= None):
        assert tgt_key_padding_mask_t is not None
        output = tgt
        hs, hs_t = [], []
        atts_all, attt_all = [], []

        for layer in self.layers:
            output, att_s = layer(output, memory, tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                output_n = self.norm(output)
                hs.append(output_n)
                atts_all.append(att_s)

        output_t = query_embed_e
        for layer in self.layers:
            output_t, att_t = layer(output_t, memory, tgt_mask=tgt_mask_t,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask_t,
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    pos=pos, query_pos=query_embed_q)
            if self.return_intermediate:
                output_n = self.norm(output_t)
                attt_all.append(att_t)
                hs_t.append(output_n)

        return paddle.stack(atts_all), paddle.stack(attt_all), paddle.stack(hs), paddle.stack(hs_t)


class TransformerEncoderLayer(paddle.nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = paddle.nn.Linear(d_model, dim_feedforward)
        self.dropout = paddle.nn.Dropout(dropout)
        self.linear2 = paddle.nn.Linear(dim_feedforward, d_model)

        self.norm1 =paddle.nn.LayerNorm(d_model)
        self.norm2 =paddle.nn.LayerNorm(d_model)
        self.dropout1 = paddle.nn.Dropout(dropout)
        self.dropout2 = paddle.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask= None,
                     src_key_padding_mask = None,
                     pos= None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos= None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask= None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayerAttMap(paddle.nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn =  paddle.nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn =  paddle.nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = paddle.nn.Linear(d_model, dim_feedforward)
        self.dropout = paddle.nn.Dropout(dropout)
        self.linear2 = paddle.nn.Linear(dim_feedforward, d_model)

        self.norm1 =paddle.nn.LayerNorm(d_model)
        self.norm2 =paddle.nn.LayerNorm(d_model)
        self.norm3 =paddle.nn.LayerNorm(d_model)
        self.dropout1 = paddle.nn.Dropout(dropout)
        self.dropout2 = paddle.nn.Dropout(dropout)
        self.dropout3 = paddle.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask= None,
                     memory_mask= None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos= None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_map = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                            key=self.with_pos_embed(memory, pos),
                                            value=memory, attn_mask=memory_mask,
                                            key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_map

    def forward_pre(self, tgt, memory,
                    tgt_mask= None,
                    memory_mask= None,
                    tgt_key_padding_mask= None,
                    memory_key_padding_mask= None,
                    pos= None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask= None,
                memory_mask= None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos= None,
                query_pos= None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return  paddle.nn.LayerList([copy.deepcopy(module) for i in range(N)])



def build_hoi_ts(args):
    return HOITransformerTS(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return paddle.nn.ReLU()
    raise RuntimeError(F"activation should be relu, not {activation}.")

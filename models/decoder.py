import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, PositionalEncoding, ScaledDotProductAttention
from .feed_forward import PoswiseFeedForwardNet
from .couplet_specific_layer import CoupletSpecificLayer
from utils import get_attn_pad_mask, get_attn_subsequence_mask
from config import (
    device,
    d_model,
    d_ff,
    d_k,
    d_v,
    n_layers,
    n_heads,
    dropout,
    GlobalVars,
    use_moe,
    num_experts,
    expert_dropout,
)

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(use_moe=use_moe, num_experts=num_experts)
        self.couplet_layer = CoupletSpecificLayer(d_model)  # 新增层

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        dec_outputs, dec_enc_attn = self.dec_enc_attn(
            dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask
        )
        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = self.couplet_layer(dec_outputs)  # 应用专用层
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        g = GlobalVars.instance()
        self.tgt_emb = nn.Embedding(g.tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = (
            self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        )
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
            device
        )  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device
        )  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0
        ).to(
            device
        )  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs
        )  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

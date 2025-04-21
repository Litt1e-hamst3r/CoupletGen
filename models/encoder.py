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

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet(use_moe=use_moe, num_experts=num_experts)
        self.couplet_layer = CoupletSpecificLayer(d_model)  # 新增层
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )
        enc_outputs = self.dropout(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        enc_outputs = self.couplet_layer(enc_outputs)  # 应用专用层
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        g = GlobalVars.instance()
        self.src_emb = nn.Embedding(g.src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

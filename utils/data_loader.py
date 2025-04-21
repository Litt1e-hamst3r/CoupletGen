import torch
import torch.utils.data as Data
import jieba
from config import (
    device,
    d_model,
    d_ff,
    d_k,
    d_v,
    n_layers,
    n_heads,
    dropout,
    tokenization_type,
)
from config import GlobalVars


def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    g = GlobalVars.instance()

    for top, bottom in sentences:
        # 根据配置选择分词方式
        if tokenization_type == 'char':
            # 每个字符作为独立Token
            top_words = list(top.strip())
            bottom_words = list(bottom.strip())
        elif tokenization_type == 'jieba':
            # 使用jieba分词（默认模式）
            top_words = list(jieba.cut(top, cut_all=False))
            bottom_words = list(jieba.cut(bottom, cut_all=False))
        elif tokenization_type == 'word':
            # 按空格分割（适用于已分好词的文本）
            top_words = top.strip().split(' ')
            bottom_words = bottom.strip().split(' ')
        else:
            raise ValueError(f"Unsupported tokenization_type: {tokenization_type}")

        # 上联处理（编码输入）
        enc_padded = top_words + ['<PAD>'] * (g.src_len - len(top_words))
        enc_input = [[g.src_vocab[word] for word in enc_padded]]

        # 下联处理（解码输入和输出）
        dec_in = ['<S>'] + bottom_words
        dec_out = bottom_words + ['<E>']
        dec_in_padded = dec_in + ['<PAD>'] * (g.tgt_len - len(dec_in))
        dec_out_padded = dec_out + ['<PAD>'] * (g.tgt_len - len(dec_out))

        dec_input = [[g.tgt_vocab[word] for word in dec_in_padded]]
        dec_output = [[g.tgt_vocab[word] for word in dec_out_padded]]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return (
        torch.LongTensor(enc_inputs),
        torch.LongTensor(dec_inputs),
        torch.LongTensor(dec_outputs),
    )


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

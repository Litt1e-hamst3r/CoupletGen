import torch
from models.transformer import Transformer
from utils.beam_search import beam_search_decoder
from utils.vocab_builder import build_vocab
from utils.data_loader import make_data, MyDataSet
from config import *
import csv
import torch.utils.data as Data
from config import GlobalVars
import pickle
import os

WEIGHTS_DIR = "weights/train_1/"
VOCAB_DIR = "vocab/train_1/"


def generate_couplet(model, top_sentence):
    g = GlobalVars.instance()

    # 生成下联（保持不变）
    dec_input = beam_search_decoder(
        model,
        enc_input=top_sentence.view(1, -1),
        start_symbol=g.tgt_vocab["<S>"],
        beam_size=3,
    )
    predict, _, _, _ = model(top_sentence.view(1, -1), dec_input)
    predict = predict.data.max(1, keepdim=True)[1]

    # 处理下联（保持不变）
    pred_list = []
    for token in predict.squeeze():
        tok = g.idx2word[token.item()]  # 目标词汇表（tgt_vocab）
        if tok == '<E>':
            break
        pred_list.append(tok)
    lower_couplet = ''.join(pred_list)

    # 处理上联：使用源词汇表（src_vocab）的逆映射
    src_idx2word = {v: k for k, v in g.src_vocab.items()}  # 动态构建逆词典
    upper_tokens = [src_idx2word[idx.item()] for idx in top_sentence]
    upper_couplet = (
        ''.join(upper_tokens).replace('<S>', '').replace('<E>', '').replace('<PAD>', '')
    )

    # 格式化输出（保持不变）
    print("上联：{}".format(upper_couplet.strip()))
    print("下联：{}".format(lower_couplet.strip()))
    print("-" * 20)


def main():
    # 加载词汇表
    with open(os.path.join(VOCAB_DIR, "src_vocab.pkl"), 'rb') as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(VOCAB_DIR, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_vocab = pickle.load(f)
    with open(os.path.join(VOCAB_DIR, 'vocab_info.pkl'), 'rb') as f:
        info = pickle.load(f)
    idx2word = info['idx2word']
    src_vocab_size = info['src_vocab_size']
    tgt_vocab_size = info['tgt_vocab_size']
    src_len = info['src_len']
    tgt_len = info['tgt_len']

    # 初始化全局变量
    g = GlobalVars.instance()
    (
        g.src_vocab,
        g.tgt_vocab,
        g.idx2word,
        g.src_vocab_size,
        g.tgt_vocab_size,
        g.src_len,
        g.tgt_len,
    ) = (
        src_vocab,
        tgt_vocab,
        idx2word,
        src_vocab_size,
        tgt_vocab_size,
        src_len,
        tgt_len,
    )

    model = Transformer().to(device)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, f"couplet_model.pth")))
    model.eval()

    sentences = []
    with open('couplet/train_mini.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 1:
                top, bottom = row[0].split(' ')
                sentences.append((top.strip(), bottom.strip()))

    enc_inputs, _, _ = next(
        iter(
            Data.DataLoader(MyDataSet(*make_data(sentences)), batch_size, shuffle=True)
        )
    )
    enc_inputs = enc_inputs.to(device)
    for i in range(len(enc_inputs)):
        top_sentence = enc_inputs[i]
        generate_couplet(model, top_sentence)


if __name__ == "__main__":
    main()

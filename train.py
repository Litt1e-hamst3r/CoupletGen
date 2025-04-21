import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.transformer import Transformer
from utils.data_loader import make_data, MyDataSet
from utils.vocab_builder import build_vocab
from config import *
import jieba
import pickle
from config import (
    device,
    learning_rate,
    batch_size,
    train_path,
    num_epochs,
    tokenization_type,
)
import csv
import os

WEIGHTS_DIR = "weights/train_1/"
VOCAB_DIR = "vocab/train_1/"


def main():
    sentences = []
    with open(train_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 1:
                top, bottom = row[0].split(' ')
                sentences.append((top.strip(), bottom.strip()))

    src_vocab, tgt_vocab, idx2word, src_vocab_size, tgt_vocab_size = build_vocab(
        sentences
    )

    src_len = 0
    tgt_len = 0

    for top, bottom in sentences:
        # 根据配置选择分词方式计算长度
        if tokenization_type == 'char':
            src_len_candidate = len(top.strip())
            tgt_len_candidate = len(bottom.strip())
        elif tokenization_type == 'jieba':
            src_len_candidate = len(list(jieba.cut(top)))
            tgt_len_candidate = len(list(jieba.cut(bottom)))
        elif tokenization_type == 'word':
            src_len_candidate = len(top.strip().split(' '))
            tgt_len_candidate = len(bottom.strip().split(' '))
        else:
            raise ValueError(f"不支持的分词类型：{tokenization_type}")

        # 更新最大长度
        if src_len_candidate > src_len:
            src_len = src_len_candidate
        if tgt_len_candidate > tgt_len:
            tgt_len = tgt_len_candidate

    # 添加 <S> 和 <E> 标记的长度
    tgt_len += 2  # 解码输入包含 <S>，输出包含 <E>，总增加 2

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

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
    loader = DataLoader(
        MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size, shuffle=True
    )

    model = Transformer().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for enc_inputs, dec_inputs, dec_outputs in tqdm(
            loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            enc_inputs, dec_inputs, dec_outputs = (
                enc_inputs.to(device),
                dec_inputs.to(device),
                dec_outputs.to(device),
            )
            outputs, _, _, _ = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        torch.cuda.empty_cache()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(VOCAB_DIR, exist_ok=True)

    # 保存模型时使用路径
    torch.save(
        model.state_dict(),
        os.path.join(WEIGHTS_DIR, f"couplet_model.pth"),
    )
    print(f"模型已保存至：{WEIGHTS_DIR}couplet_model.pth")

    # 保存词汇表
    with open(os.path.join(VOCAB_DIR, "src_vocab.pkl"), "wb") as f:
        pickle.dump(src_vocab, f)
    with open(os.path.join(VOCAB_DIR, "tgt_vocab.pkl"), "wb") as f:
        pickle.dump(tgt_vocab, f)
    with open(os.path.join(VOCAB_DIR, "vocab_info.pkl"), "wb") as f:
        pickle.dump(
            {
                'idx2word': idx2word,
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
                'src_len': src_len,
                'tgt_len': tgt_len,
            },
            f,
        )
    print(f"词汇表已保存至：{VOCAB_DIR}")


if __name__ == "__main__":
    main()

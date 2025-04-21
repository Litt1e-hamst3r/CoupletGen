import torch
from models.transformer import Transformer
from utils.beam_search import beam_search_decoder
from config import device, GlobalVars
import jieba
import os
import pickle
from config import tokenization_type

WEIGHTS_DIR = "weights/train_1/"
VOCAB_DIR = "vocab/train_1/"


def generate_couplet(model, top_sentence):
    g = GlobalVars.instance()

    dec_input = beam_search_decoder(
        model,
        enc_input=top_sentence.view(1, -1),
        start_symbol=g.tgt_vocab["<S>"],
        beam_size=3,
    )
    predict, _, _, _ = model(top_sentence.view(1, -1), dec_input)
    predict = predict.data.max(1, keepdim=True)[1]

    pred_list = []
    for token in predict.squeeze():
        tok = g.idx2word.get(token.item(), '<UNK>')  # 处理未知词
        if tok in ('<E>', '<PAD>'):
            break
        pred_list.append(tok)
    lower_couplet = ''.join(pred_list)

    # 处理上联显示（用户输入的原始文本）
    upper_couplet = user_input.strip()  # 直接使用原始输入文本

    print("上联：{}".format(upper_couplet))
    print("下联：{}".format(lower_couplet))
    print("-" * 20)


def preprocess_input(user_input):
    g = GlobalVars.instance()
    # 根据配置选择分词方式
    if tokenization_type == 'char':
        words = list(user_input.strip())  # 每个字符独立
    elif tokenization_type == 'jieba':
        words = list(jieba.cut(user_input.strip(), cut_all=False))
    elif tokenization_type == 'word':
        words = user_input.strip().split(' ')
    else:
        raise ValueError(f"不支持的分词类型：{tokenization_type}")

    # 填充到固定长度
    padded = words + ['<PAD>'] * (g.src_len - len(words))

    # 转换为索引，处理未知词
    indices = []
    unk_idx = g.src_vocab.get('<UNK>', 0)  # 获取未知词标记的索引
    for word in padded:
        # 优先使用词汇表中的索引，否则使用<unk>
        indices.append(g.src_vocab.get(word, unk_idx))

    return torch.LongTensor([indices]).to(device)


def main():
    # 加载词汇表和模型
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

    g = GlobalVars.instance()
    g.src_vocab = src_vocab
    g.tgt_vocab = tgt_vocab
    g.idx2word = idx2word
    g.src_vocab_size = src_vocab_size
    g.tgt_vocab_size = tgt_vocab_size
    g.src_len = src_len
    g.tgt_len = tgt_len

    model = Transformer().to(device)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "couplet_model.pth")))
    model.eval()

    # 用户交互循环
    while True:
        global user_input
        user_input = input("请输入上联（输入 '退出' 结束程序）：")
        if user_input.strip().lower() == '退出':
            print("程序已退出。")
            break

        try:
            input_tensor = preprocess_input(user_input)
            generate_couplet(model, input_tensor)
        except KeyError as e:
            print(f"输入包含未知词汇：{e}，请使用训练数据中的词汇。")
        except Exception as e:
            print(f"生成失败：{str(e)}")


if __name__ == "__main__":
    main()

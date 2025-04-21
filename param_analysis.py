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

    # 分析模型参数
    print("\n模型参数分析:")
    total_params = 0
    ffn_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            if 'pos_ffn' in name:  # FFN层参数
                ffn_params += num_params
            print(f"{name}: {num_params:,} 参数")

    print(f"\n总参数: {total_params:,}")
    print(f"FFN层参数: {ffn_params:,} (占比: {ffn_params/total_params*100:.1f}%)")
    print("-" * 50 + "\n")

    # 参数分布可视化
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # 设置中文字体
    try:
        font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf', size=12)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass  # 如果找不到字体则使用默认字体

    other_params = total_params - ffn_params
    labels = ['FFN层', '其他层']
    sizes = [ffn_params, other_params]
    colors = ['#ff9999', '#66b3ff']

    plt.figure(figsize=(8, 6))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
    )
    plt.title('Transformer模型参数分布')
    plt.axis('equal')
    plt.savefig('img/param_distribution.png')
    print("已生成参数分布图: img/param_distribution.png")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()

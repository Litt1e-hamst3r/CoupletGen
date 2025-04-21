import jieba
from config import tokenization_type
def build_vocab(sentences):
    src_words = set()
    tgt_words = set()

    for top, bottom in sentences:
        # 根据配置选择分词方式
        if tokenization_type == 'char':
            src_tokens = list(top.strip())
            tgt_tokens = list(bottom.strip())
        elif tokenization_type == 'jieba':
            src_tokens = list(jieba.cut(top, cut_all=False))
            tgt_tokens = list(jieba.cut(bottom, cut_all=False))
        elif tokenization_type == 'word':
            src_tokens = top.strip().split(' ')
            tgt_tokens = bottom.strip().split(' ')
        else:
            raise ValueError(f"不支持的分词类型：{tokenization_type}")

        src_words.update(src_tokens)
        tgt_words.update(tgt_tokens)

    src_words.add('<PAD>')
    tgt_words.update(['<PAD>', '<S>', '<E>'])

    src_vocab = {word: i for i, word in enumerate(src_words)}
    tgt_vocab = {word: i for i, word in enumerate(tgt_words)}
    idx2word = {v: k for k, v in tgt_vocab.items()}

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    return src_vocab, tgt_vocab, idx2word, src_vocab_size, tgt_vocab_size

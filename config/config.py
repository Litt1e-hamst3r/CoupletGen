import torch

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = 256
d_ff = 2048
d_k = d_v = 64
n_layers = 4
n_heads = 4
dropout = 0.1
num_epochs = 10
learning_rate = 1e-4
batch_size = 128
train_path = 'couplet/train.csv'
tokenization_type = 'char'

# Moe参数
use_moe = True
num_experts = 4
expert_dropout = 0.1 
moe_k = 2  # Number of experts to use per input
capacity_factor = 1.0  # Capacity buffer for each expert
noise_epsilon = 1e-2  # Noise scale for gating

# 全局变量管理类
class GlobalVars:
    _instance = None
    
    def __init__(self):
        self.src_vocab = None
        self.tgt_vocab = None
        self.idx2word = None 
        self.src_vocab_size = 0
        self.tgt_vocab_size = 0
        self.src_len = 0
        self.tgt_len = 0
        
    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = GlobalVars()
        return cls._instance

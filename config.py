import torch
import torch.nn.functional as F

"""
special tokens
"""
PAD = 0
START = 3
END = 102
UNK = 1
MASK = 103
"""
hyper-parameters
"""
enc_limit =100#maximum limit length of a sequence in a batch
dec_limit =90#maximum limit length of a sequence in a batch
nonlinear = torch.tanh
MAX_CLIP = 2
N_GPU = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#hyper-parameters
######################
h_size=256#hidden size of encoder
d_size=512#hidden size of decoder
w_size=300#project size of attention
b_size=32#batch size
kenum = 6#k number
lr=0.001#learning rate
lazy_step = 1#step for gradient accumulation
thresh = 0.95#threshold for point mode
data_dir = './dataset/cnn-dailymail'#dataset directory
embed= data_dir + '/bertglove.mat'#pre-trained word embedding
check_dir = './seq2seq/cnn/attn'#check point file
######################

import model_baseline
import math
import transformers
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import re
from transformers import BertModel
from transformers import PreTrainedTokenizerFast
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
from torch.nn import LSTM, Softmax
from torch.nn.utils.rnn import pad_sequence
from einops import repeat, rearrange
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from tqdm import tqdm
import data
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
#from slr import preprocess
import numpy as np




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class SLRWeightModel(nn.Module):
    def __init__(self, num_classes, d_model, nhead, seq_len, nlayers, mask= True):
        super(SLRWeightModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.mask = mask
        self.hidden = 2*self.d_model
        self.weights = nn.Parameter(torch.ones(1, 1, self.d_model, 1)) # 2 -> 1
        self.src_emb = nn.Linear(2*self.d_model, self.hidden)
        encoder_layer = nn.TransformerEncoderLayer(self.hidden, nhead, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos_encoding = PositionalEncoding(self.hidden, max_len=seq_len)
        self.linear = nn.Linear(self.hidden, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 2*self.d_model))
        
        

    def forward(self, src, src_padding_mask): 
        # print(src.shape) 16 100 137 2
        
        b, f, v, c = src.shape
        # print(b, f, v, c) # 16 100 137 2
        src = src * torch.nn.functional.softmax(self.weights, dim=2) * v
        #src = src * self.weights
        src = src.view(b, f, v*c)
        #k, n, m = src.shape # 16 100 274
        #src = self.src_emb(src)
        #cls_tokens = repeat(self.cls_token, '1 1 m -> k 1 m', k = k )
        #h = torch.cat((cls_tokens, src), dim=1)
        h = src
        

        h = self.encoder(src =  h, src_key_padding_mask = src_padding_mask)
        h = h.mean(dim=1)
        res = self.linear(h)
        return res
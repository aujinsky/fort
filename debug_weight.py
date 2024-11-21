import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchmetrics
from model_baseline_slr import SLRModel
from model_baseline_slr_weight import SLRWeightModel
from model_baseline_slr_encoderdecoder import SLREncoderDecoderModel
from model_baseline_slr_spoter import SLRSPOTERModel
from model_baseline_slr_encoderdecoder_weight import SLREncoderDecoderWeightModel
from model_baseline_slr_spoter_weight import SLRSPOTERWeightModel
from data import NiaSLDataset20, WlaSLDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn.utils.rnn import pad_sequence
#from pytorch_lightning.profiler import AdvancedProfiler, PyTorchProfiler
#from argparse import \
from torchtext.vocab import vocab
import argparse
import torch.optim
from slr import transformation
import random
import os


import main_slr_multi_lmdb_wlasl

    

if __name__=="__main__":
    dataset_type = "wlasl"
    random.seed(42)
    torch.manual_seed(42)

    if dataset_type == "wlasl":
        with open('/home/ajkim/kslt/slr/WLASL100_GLOSSES.json','r') as f:
            dict_json = json.load(f)
            word_vocab = vocab(dict_json, specials=["<unk>"])
            word_dict = word_vocab.get_stoi()
            classes = list(word_dict)
            num_classes = len(classes)
            print("num_classes: %d"%num_classes)
    elif dataset_type == "niasl":
        with open('/home/ajkim/kslt/slr/NIA_classes3.json','r') as f:
            dict_json = json.load(f)
            word_dict = dict_json[0].keys()
            classes = list(word_dict)
            num_classes = len(classes)
            print("num_classes: %d"%num_classes)
        
    #profiler1 = AdvancedProfiler(filename="profiler_output")
    #profiler2 = PyTorchProfiler(filename="profiler_pytorch")
    # 0.5
    ckpt = "/home/ajkim/kslt/slr_logs/wlasl/version_291/checkpoints/weight-()-v_num=0-epoch=174-val_acc_step=0.624.ckpt"
    # hand
    ckpt = "/home/ajkim/kslt/slr_logs/wlasl/version_528/checkpoints/weight_augment-()-v_num=0-epoch=15-val_acc_step=0.615.ckpt"
    model = main_slr_multi_lmdb_wlasl.TrainerModule.load_from_checkpoint(ckpt, aug_method = [], classes = classes, dataset = dataset_type, model = "weight_augment", body12=True)
    
    import IPython; IPython.embed(); exit(1)

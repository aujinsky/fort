import glob
import gzip
import os
import pickle
import random
from einops import rearrange
import json
import xml.etree.ElementTree as ET
import lmdb
import typing as tp
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.vocab import vocab
from torch.utils.data import Dataset
from collections import defaultdict
from slr import transformation
from collections import Counter
#from utils import load_dirs, load_json
import matplotlib.pyplot as plt
#from pose_format.pose import Pose
#from pose_format.numpy import NumPyPoseBody
from torch.nn.utils.rnn import pad_sequence
#from render import KeypointRenderer
#from pose_format.pose_visualizer import PoseVisualizer
from transformers import AutoTokenizer, AutoModelWithLMHead


def get_random_seq(seq, seq_len):
    start = random.randrange(0, len(seq) + 1 - seq_len)
    end = start + seq_len
    return seq[start : end]


def load_data(
    dataset_type, 
    train_trans_path = None, 
    valid_trans_path = None, 
    test_trans_path = None, 
    seq_len = -1, 
    min_seq_len = -1,
    normalize = False
):
    assert dataset_type in ['phoenix', 'how2sign', 'niasl21'], 'dataset must be selected between phoenix or how2sign.'
    
    if dataset_type == 'how2sign':
        root, _ = os.path.split(train_trans_path)

        # define joint paths
        tr_joint_path = os.path.join(root, 'train_2D_keypoints', 'json')
        val_joint_path = os.path.join(root, 'val_2D_keypoints', 'json')
        tst_joint_path = os.path.join(root, 'test_2D_keypoints', 'json')

        # tr_joint_feat_path = os.path.join(root, 'train_feat')
        # val_joint_feat_path = os.path.join(root, 'val_feat')
        # tst_joint_feat_path = os.path.join(root, 'val_feat')

        trainset = How2SignDataset(
            trans_path = train_trans_path, 
            joint_path = tr_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        
        validset = How2SignDataset(
            trans_path = valid_trans_path, 
            joint_path = val_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )

        testset = How2SignDataset(
            trans_path = test_trans_path, 
            joint_path = tst_joint_path, 
            seq_len = seq_len, 
            min_seq_len = min_seq_len,
            normalize = normalize
        )
    elif dataset_type == 'phoenix':
        trainset = Phoenix2014TDataset(
            db_path = train_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )
        validset = Phoenix2014TDataset(
            db_path = valid_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )
        
        testset = Phoenix2014TDataset(
            db_path = test_trans_path, 
            min_seq_len = min_seq_len, 
            seq_len = seq_len,
            normalize = normalize
        )
    elif dataset_type == 'niasl21':
        trainset = NiaSLDataset(
            fpath = train_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        validset = NiaSLDataset(
            fpath = valid_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
        testset = NiaSLDataset(
            fpath = test_trans_path,
            seq_len = seq_len,
            min_seq_len = min_seq_len,
            normalize = normalize
        )
    else:
        raise NotImplementedError
    
    return trainset, validset, testset



class How2SignDataset(Dataset):
    def __init__(
        self, 
        trans_path, 
        joint_path,
        seq_len = -1, 
        min_seq_len = -1,
        normalize = True
    ):
        super().__init__()

        data = self._load_df(
            dataframe = pd.read_csv(trans_path, sep='\t'), 
            joint_path = joint_path, 
            min_seq_len = min_seq_len
        )
            
        self.max_seq_len = data.FRAME_LENGTH.max()
        self.min_seq_len = data.FRAME_LENGTH.min()
        
        self.seq_len = seq_len
        self.min_len = min_seq_len
        self.joint_path = joint_path
        self.trans_path = trans_path

        self.data = data

        self.normalize = normalize
        
    def _load_df(self, dataframe, joint_path, min_seq_len):
        joint_dir_list = []
        frame_len_list = []
        
        for i in range(len(dataframe)):
            data = dataframe.iloc[i]
            skel_id = data['SENTENCE_NAME']
            skel_dir = os.path.join(joint_path, skel_id)
            frame_len = len(glob.glob(os.path.join(skel_dir, '*')))
            joint_dir_list.append(skel_dir)
            frame_len_list.append(frame_len)
        
        dataframe['JOINT_DIR'] = joint_dir_list
        dataframe['FRAME_LENGTH'] = frame_len_list

        if min_seq_len != -1:
            dataframe.drop(dataframe[dataframe.FRAME_LENGTH < min_seq_len].index, inplace = True)

        return dataframe

    def _load_skeleton(self, signs):
        POSE_IDX = 8 # Upper body pose
        skel_list = []
        
        for l in signs:
            json_file = l['people'][0]
            
            pose = json_file['pose_keypoints_2d']
            del pose[2::3]
            del pose[POSE_IDX*2:]
            
            landmark = json_file['face_keypoints_2d']
            del landmark[2::3]
            
            right_hand = json_file['hand_right_keypoints_2d']
            del right_hand[2::3]
            
            left_hand = json_file['hand_left_keypoints_2d']
            del left_hand[2::3]
            
            skel_list += [pose + right_hand + left_hand + landmark]
        
        return torch.Tensor(skel_list)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        
        id = data['SENTENCE_NAME']
        text = data['SENTENCE']
        joint_dir = data['JOINT_DIR']
        frame_len = data['FRAME_LENGTH']
        
        #joint_dirs = load_dirs(os.path.join(joint_dir, '*'))
        #joint_feats = self._load_skeleton([load_json(jd) for jd in joint_dirs])
        
        if self.seq_len != -1:
            joint_feats = get_random_seq(joint_feats, self.seq_len)
            frame_len = len(joint_feats)
                
        joint_feats = rearrange(joint_feats, 't (v c) -> t v c', c = 2)
        t, v, c = joint_feats.size()
        
        if self.normalize:
            dist = (joint_feats[:, 2, :] - joint_feats[:, 5, :]).pow(2).sum(-1).sqrt()
            joint_feats /= (dist / 300).view(dist.size(0), 1, 1).repeat(1, v, c)
            
            center = torch.ones(joint_feats[:, 1, :].size())
            center[:, 0] *= (1400 // 2)
            center[:, 1] *= (1050 // 2)
            joint_feats -= (joint_feats[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

            joint_feats[:, :, :1] /= (1400 * 2.5)
            joint_feats[:, :, 1:] /= (1050 * 2.5)

            joint_feats += 0.3 # numeric stable

        return {
            'id': id,
            'text': text,
            'joint_feats': joint_feats,
            'frame_len': frame_len
        }
        
    def __len__(self):
        return len(self.data)

def extend_tens(skel, f):
  n = skel.shape[0]
  temp = float(1/n)
  sixty = float(1/f)
  ret = []
  for i in range(1, f+1):
    num = int(sixty*i/temp)
    if(num == n):
      num = n-1
    ret.append(skel[num].view(1, -1, 2))
  return_tens = torch.stack(ret)  
  return_tens = return_tens.view(f, -1, 2)
  return return_tens

def shorten_list(skel, rate):
    n = len(skel)
    temp = int(n/rate)
    ret = []
    for i in range(temp):
        ret.append(skel[i*rate])
    ret = torch.stack(ret)
    return ret

def normalize_2d_keypoints(keypoints_2d, width, height): # 1920, 1080
    normalized_keypoints_2d = keypoints_2d.clone()
    scale = torch.tensor([width, height], device=keypoints_2d.device)
    return normalized_keypoints_2d / scale

def normalize_skeleton_2d(X, resize_factor=None):
    def distance_2d(x1, y1, x2, y2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 2)  # neck

    if resize_factor is None:
        neck_height = distance_2d(X[:, 1, 0], X[:, 1, 1],
                                  X[:, 0, 0], X[:, 0, 1]).float()
        shoulder_length = distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 2, 0], X[:, 2, 1]) + \
                          distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 5, 0], X[:, 5, 1])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length
        
        for i in range(len(resize_factor)):
            if resize_factor[i] == 0:
                if i>0:
                    resize_factor[i] = resize_factor[i-1]
                else:
                    resize_factor[i] = resize_factor[i+1]


    normalized_X = X.clone()
    
    
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor

    return normalized_X
def normalize(X):
    """
    Normalize 3D keypoints using min-max normalization.

    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints

    Returns:
        X_norm: torch tensor of shape (num_frames, num_joints, 3) representing the normalized 3D keypoints
    """
    T, n, d = X.shape
    X = X.reshape(T*n, d)
    X_min = torch.min(X, dim=0)[0]
    X_max = torch.max(X, dim=0)[0]
    X_norm = -1.0 + 2.0 * (X - X_min) / (X_max - X_min)
    X_norm = X_norm.reshape(T, n, d)
    
    return X_norm

def normalize_skeleton_2d_mod(X, resize_factor=None):
    def distance_2d(x1, y1, x2, y2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 2)  # neck

    if resize_factor is None:
        neck_average = X.mean(dim=1)
        neck_height = distance_2d(X[:, 1, 0], X[:, 1, 1],
                                  X[:, 0, 0], X[:, 0, 1]).float()
        shoulder_length = distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 2, 0], X[:, 2, 1]) + \
                          distance_2d(X[:, 1, 0], X[:, 1, 1], X[:, 5, 0], X[:, 5, 1])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length
        
        for i in range(len(resize_factor)):
            if resize_factor[i] == 0:
                if i>0:
                    resize_factor[i] = resize_factor[i-1]
                else:
                    resize_factor[i] = resize_factor[i+1]


    normalized_X = X.clone()
    
    
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor

    return normalized_X

class WMTSLTDataset(Dataset):
    def __init__(self, fold_path, data_type, seq_len = -1, min_seq_len = -1, normalize = True, padding = False, sub_sample = False):
        self.seq_len = seq_len
        self.fold_path = fold_path
        self.data_type = data_type
        self.normalize = normalize
        self.maxx = 0
        self.maxy = 0
        self.padding = padding
        self.sub_sample = sub_sample
        self._load_data(fold_path)
        csv_file = pd.read_csv(filepath_or_buffer = '/data/sl_datasets/wmt_slt/metadata_train.csv', sep=',')
        self.labels = dict(csv_file[['id', 'example']].values)
        
    def _load_data(self, fold_path):
        self.data_path = []
        self.index = []
        if self.data_type == "openpose":
            for filename in glob.glob(fold_path+'/ex_openpose/example_openpose/*'):
                self.data_path.append(filename)
                self.index.append(filename.split('.')[0].split('/')[-1])
        elif self.data_type == "mediapipe":
            for filename in glob.glob(fold_path+'/ex_mediapipe/example_mediapipe/*'):
                self.data_path.append(filename)
                self.index.append(filename.split('.')[0].split('/')[-1])

    def __getitem__(self, index):
        file_name = self.data_path[index]
        ind = self.index[index]
        buffer = open(file_name, "rb").read()
        p = Pose.read(buffer, NumPyPoseBody)
        full = p.torch().body.data.tensor
        if self.data_type == "openpose":
            full = full.view(-1, 137, 2)
        else:
            full = full.view(-1, 543, 3)
        #full = shorten_list(full, 3)
        framel = full.shape[0]
        if self.normalize == True:
            full = transformation.preprocess_keypoints(full)
        right = full[:,:,95:116]
        left = full[:,:,116:]
        body = full[:,:,:18]
        face = full[:,:,18:95]
        label = self.labels[int(ind)]
        return {'input' : full, 'right' : right, 'left' : left, 'body' : body, 'face' : face, 'index' : ind, 'label' : label, 'frame_lengths' : framel, 'gloss' : ''}
            
    def __len__(self):
        return len(self.data_path)

class NiaSLDataset20(Dataset):
    def __init__(self, db_path, seq_len = -1, min_seq_len = -1, normalize = True, padding = False, sub_sample = False, crop = False, normalize_hand = False):
        super().__init__()
        self.seq_len = seq_len
        self.normalize = normalize
        self.padding = padding
        self.sub_sample = sub_sample
        self.db_path = db_path
        self.crop = crop
        self.normalize_hand = normalize_hand
        self.env = lmdb.open(self.db_path, readonly = True, lock = False, readahead= False, meminit = False)

        self.txn = self.env.begin()
        self.len = self.env.stat()['entries']

        with open('/home/ajkim/kslt/slr/NIA_classes3.json','r') as f:
            dict_json = json.load(f)
            word_dict = dict_json[0]
            word_dict = defaultdict(int,word_dict)
        self.dict = word_dict
        
    
    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError()
        data = self.txn.get(str(index).encode("utf-8"))
        data = pickle.loads(data)
        text = data["text"]
        gloss = data["gloss"]
        tensor = data["keypoint"].squeeze()
        index = data["index"]
        tensor = tensor.view(-1, tensor.shape[1], 2)
        if self.crop == True:
            l = tensor.shape[0]
            start = 0
            end = l-1
            for i in range(l):
                if i < l-5:
                    X_diff = torch.diff(tensor[[i, i+5]], dim = 0)
                    distances = torch.norm(X_diff, dim=-1)
                    distances = torch.mean(distances, dim=-1)
                    if distances.item() > 6:
                        start = i
                        break
            tensor = tensor[start:]
        
        if self.normalize == True:
            #mod_tens = normalize_2d_keypoints(mod_tens, 1920, 1280)
            tensor = transformation.preprocess_keypoints(tensor, norm = 'norm', remove_noisy = False, normalize_hand = self.normalize_hand) #normalize_skeleton_2d(mod_tens)
            #mod_tens = normalize(mod_tens)
        if "dataset_type" in data:
            if data["dataset_type"] == "rsz":
                tensor = 0.9 * tensor
        if self.sub_sample == True:
            tensor = extend_tens(tensor, 100)
        t, f, d = tensor.shape
        frame_length = t
        if self.padding == True:
            zeros = torch.zeros(512, f, d)
            zeros[:t, :, :] = tensor
            tensor = zeros
        return {'text' : text, 'gloss' : gloss, 'keypoint' : tensor, "index" : index, 'len_list' : frame_length}

    def __len__(self):
        return self.len


class WlaSLDataset(Dataset):
    def __init__(self, db_path, seq_len = -1, min_seq_len = -1, normalize = True, padding = True, sub_sample = False, extended_length = 100, normalize_hand = True):
        super().__init__()
        self.seq_len = seq_len
        self.normalize = normalize
        self.padding = padding
        self.sub_sample = sub_sample
        self.db_path = db_path
        self.extended_length = extended_length
        self.env = lmdb.open(self.db_path, readonly = True, lock = False, readahead= False, meminit = False)
        self.txn = self.env.begin()
        self.len = self.env.stat()['entries']
        self.normalize_hand = normalize_hand
        """
        with open('/home/ajkim/kslt/slr/WLASL_GLOSSES.json','r') as f:
            dict_json = json.load(f)
            self.dict = vocab(dict_json, specials=["<s>", "</s>", "<usr>", "<pad>", "<sys>", "<unk>", "<mask>"])
            self.dict.set_default_index(5)      
        self.dict = self.dict.get_stoi()
        """
    
    def __getitem__(self, index):
        data = self.txn.get(str(index).encode("utf-8"))
        if index >= self.len:
            raise IndexError()
        if data == None:
            return {'text' : "", 'gloss' : "", 'keypoint' : torch.zeros((0, 67, 2)), "index" : ""}
        data = pickle.loads(data)
        text = data["text"]
        gloss = data["gloss"]
        if data["keypoint"].dim() == 3:
            b, d, e = data["keypoint"].shape
            tensor = data["keypoint"].view(b, d, e)
        elif data["keypoint"].dim() == 2:
            b, d = data["keypoint"].shape
            tensor = data["keypoint"].view(b, d//2, 2)
        index = data["index"]
        if self.sub_sample == True:
            tensor = extend_tens(tensor, self.extended_length)
        if self.normalize == True:
            #mod_tens = normalize_2d_keypoints(mod_tens, 1920, 1280)
            tensor = transformation.preprocess_keypoints(tensor, norm = 'norm', remove_noisy = False, normalize_hand = self.normalize_hand) #normalize_skeleton_2d(mod_tens)
            #mod_tens = normalize(mod_tens)
        t, f, d = tensor.shape
        if self.padding == True:
            zeros = torch.zeros(512, f, d)
            zeros[:t, :, :] = tensor
            tensor = zeros
        frame_length = t
        if not '2x' in self.db_path:
            return {'text' : text, 'gloss' : gloss, 'keypoint' : tensor, "index" : index}
        else:
            return {'text' : text, 'gloss' : gloss, 'keypoint' : tensor, "index" : data["seq_index"]}

    def __len__(self):
        return self.len

class Phoenix2014TDataset(Dataset):
    def __init__(self, db_path, seq_len = -1, min_seq_len = -1, normalize = True, padding = False, sub_sample = False, crop = False):
        super().__init__()
        print("db_path:", db_path)
        self.seq_len = seq_len
        # filtered by a given length
        if min_seq_len != -1:
            self.data = [data for data in self.data if len(data['sign']) > min_seq_len]
        self.normalize = normalize
        self.padding = padding
        self.sub_sample = sub_sample
        self.db_path = db_path
        self.crop = crop
        self.env = lmdb.open(self.db_path, readonly = True, lock = False, readahead= False, meminit = False)
        self.txn = self.env.begin()
        self.len = self.env.stat()['entries']
        with open('/home/ajkim/kslt/slr/PHOENIX_14_GLOSSES.json','r') as f:
            dict_json = json.load(f)
            self.dict = vocab(dict_json, specials=["<s>", "</s>", "<usr>", "<pad>", "<sys>", "<unk>", "<mask>"])
            self.dict.set_default_index(5)      
        self.len = self.env.stat()['entries']  
        
        
    
    def __getitem__(self, index):
        data = self.txn.get(str(index).encode("utf-8"))
        data = pickle.loads(data)
        text = data["text"]
        gloss = data["gloss"]
        tensor = data["joint_feats"]
        index = data["id"]
        joint = tensor.squeeze()

        if self.normalize:
            if self.seq_len != -1:
                joint = get_random_seq(joint, self.seq_len)
        
        
            pose = joint[:, :8, :]
            landmark = joint[:, 50:, :]
        
            landmark *= 0.4

            neck = pose[:, 0]
            nose = landmark[:, 30]

            diff = neck - nose
            diff = rearrange(diff, 't v -> t () v').repeat(1, landmark.size(1), 1)
            landmark += diff

            joint_feats = rearrange(joint, 't v c -> t v c')
            t, v, c = joint_feats.size()
            dist = (joint_feats[:, 2, :] - joint_feats[:, 5, :]).pow(2).sum(-1).sqrt()
            joint_feats /= (dist / 0.3).view(dist.size(0), 1, 1).repeat(1, v, c)

            center = torch.ones(joint_feats[:, 1, :].size()) * 0.5
            joint_feats -= (joint_feats[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

            joint_feats[:, :, :1] /= 1.6
            joint_feats[:, :, 1:] /= 1.6

            joint_feats += 0.1
        else:
            joint_feats = tensor.squeeze()
        return {
            'id': id,
            'text': text,
            'gloss': gloss,
            'joint_feats': joint_feats,
            'frame_len': len(joint_feats)
        }

    def __len__(self):
        return self.len
    def collate_fn(self, batch):
        feats_batch, gloss_batch, text_batch, len_batch = [], [], [], []
        id_batch = []
        for item in batch:
            if item == None:
                import IPython; IPython.embed(); exit(1)
    
            feats_batch.append(item['joint_feats'])
            gloss_batch.append(item['gloss'])
            if isinstance(item['text'], list):
                item['text'] = ''.join(item['text'])
                # print(item['text'])
            text_batch.append("<s> " + item['text'] + " </s>")
            id_batch.append(item['id'])
            len_batch.append(item['frame_len'])
        
        feats_batch = pad_sequence(feats_batch)
        feats_batch = torch.reshape(feats_batch, (feats_batch.shape[0], feats_batch.shape[1], feats_batch.shape[2]*feats_batch.shape[3]))
        
        return feats_batch, gloss_batch, text_batch, id_batch, len_batch


class NiaSLDataset(Dataset):
    def __init__(self, fpath, min_seq_len = -1, seq_len = -1, normalize = True):
        super().__init__()

        self.path_len = len(fpath)

        self.seq_len = seq_len
        self.data = self._load_data(fpath)

        self.normalize = normalize

    def _load_data(self, fpath):
        data_id = []
        kor_text = []
        joint_f = []
        gloss = []
        count = 0
        self.data_path = []
        for filename in glob.glob(fpath+'/*/*/*/*/*'):
            self.data_path.append(filename)
            
            
    
    def _load_skeleton(self, left_l, face_l, pose_l, right_l):
        POSE_IDX = 8 # Upper body pose
        skel_list = []
        
        for left_hand, landmark, pose, right_hand in zip(left_l, face_l, pose_l, right_l):
        
            del pose[2::3]
            del pose[POSE_IDX*2:]
            del landmark[2::3]
            del right_hand[2::3]
            del left_hand[2::3]
            
            skel_list += [pose + right_hand + left_hand + landmark]
        
        return torch.Tensor(skel_list)

        
    def __getitem__(self, index):
        '''
        {'id': '25October_2010_Monday_tagesschau-17',
        'text': 'regen und schnee lassen an den alpen in der nacht nach im norden und nordosten fallen hier und da schauer sonst ist das klar .',
        'gloss': 'REGEN SCHNEE REGION VERSCHWINDEN NORD REGEN KOENNEN REGION STERN KOENNEN SEHEN',
        'joint_feats': tensor([[[0.4078, 0.2775],
                [0.4125, 0.4125],
                [0.3190, 0.4292],
                ...,,
        'frame_len': 181}
        '''
        filename = self.data_path[index]
        xml_file_path = "/data/sl_datasets/niasl21/original_test/labeled_data/xml_files" + filename[self.path_len:-4].replace('morpheme', 'keypoints') + "xml"
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        with open(filename, 'r') as file:
            for a in root.findall('size'):
                print(a.attrib)
            data = json.load(file)
            id = data["metadata"]["id"]
            text = data["korean_text"]
            gloss_list = []

            #get gloss sentence
            for gloss_cat in data["sign_script"]:
                for gloss_json in data["sign_script"][gloss_cat]:
                    gloss_list.append([gloss_json["gloss_id"], gloss_json["start"]])
            gloss_list.sort(key=lambda x: x[1])
            gloss_sent = ""
            for gloss_elem in gloss_list:
                gloss_sent += gloss_elem[0] + " "

            frame_len = int(root[1][0].find('size').text)
            left = []
            face = []
            right = []
            pose = []
            for i in range(frame_len):
                seq = root[4*i +2][0].get('points')
                left_elem = seq.replace(";",',').split(',')
                left_elem = list(map(float, left_elem))
                left.append(left_elem)

                seq = root[4*i +3][0].get('points')
                face_elem = seq.replace(";",',').split(',')
                face_elem = list(map(float, face_elem))
                face.append(face_elem)

                seq = root[4*i +4][0].get('points')
                pose_elem = seq.replace(";",',').split(',')
                pose_elem = list(map(float, pose_elem))
                pose.append(pose_elem)

                seq = root[4*i +5][0].get('points')
                right_elem = seq.replace(";",',').split(',')
                right_elem = list(map(float, right_elem))
                right.append(right_elem)
            
            joint_feats = self._load_skeleton(left, face, pose, right)

            if self.seq_len != -1:
                joint_feats = get_random_seq(joint_feats, self.seq_len)
                frame_len = len(joint_feats)
                    
            joint_feats = rearrange(joint_feats, 't (v c) -> t v c', c = 2)
            t, v, c = joint_feats.size()
            
            if self.normalize:
                dist = (joint_feats[:, 2, :] - joint_feats[:, 5, :]).pow(2).sum(-1).sqrt()
                joint_feats /= (dist / 300).view(dist.size(0), 1, 1).repeat(1, v, c)
                
                center = torch.ones(joint_feats[:, 1, :].size())
                center[:, 0] *= (1400 // 2)
                center[:, 1] *= (1050 // 2)
                joint_feats -= (joint_feats[:, 1, :] - center).unsqueeze(1).repeat(1, v, 1)

                joint_feats[:, :, :1] /= (1400 * 2.5)
                joint_feats[:, :, 1:] /= (1050 * 2.5)

                joint_feats += 0.3 # numeric stable

        return {
            'id': id,
            'text': text,
            'joint_feats': joint_feats,
            'frame_len': frame_len,
            
        }

    def __len__(self):
        return len(self.data)


class AutSLDataset(Dataset):
    def __init__(self, db_path, seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, crop = False):
        super().__init__()
        self.seq_len = seq_len
        self.normalize = normalize
        self.padding = padding
        self.sub_sample = sub_sample
        self.db_path = db_path
        self.crop = crop
        self.env = lmdb.open(self.db_path, readonly = True, lock = False, readahead= False, meminit = False)
        self.txn = self.env.begin()
        self.len = self.env.stat()['entries']
        
    def __getitem__(self, index):
        data = self.txn.get(str(index).encode("utf-8"))
        data = pickle.loads(data)
        text = data["text"]
        gloss = data["gloss"]
        tensor = data["keypoint"]
        index = data["index"]

        if self.sub_sample == True:
            tensor = extend_tens(tensor, 100)
        t, f, d = tensor.shape
        frame_length = t
        if self.padding == True:
            zeros = torch.zeros(512, f, d)
            zeros[:t, :, :] = tensor
            tensor = zeros
        return {'text' : text, 'gloss' : gloss, 'keypoint' : tensor, "index" : index, 'len_list' : frame_length}

    def __len__(self):
        return self.len 

if __name__ == '__main__':
    '''
    sample runnable code,
    you can check the outputs
    '''

    '''dataset_wmt = WMTSLTDataset(
        fold_path = '/data/sl_datasets/wmt_slt/train',
        data_type = 'mediapipe',
        normalize = False
    )
    dataset_wmt_open = WMTSLTDataset(
        fold_path = '/data/sl_datasets/wmt_slt/train',
        data_type = 'openpose',
        normalize = False
    )

    renderer = KeypointRenderer()
    data1 = dataset_wmt[1]['input'][:]
    count = 0
    data2 = dataset_wmt[2]['input'][:]
    print(data2[0,-42:])
    #renderer.save_3d_animation(reference = data2, generated = data1)
    for dat in data1:
        x = dat[:,0]
        y = dat[:,1]
        print(x)
        print(y)
        print(x.shape)
        print(y.shape)
        plt.scatter(x,y)
        plt.savefig('foo.png')
        break

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
    tokens = tokenizer(text= dataset_wmt[1]['label'])
    token = tokenizer(text = '<pad>')
    print(tokens)
    print('hello')
    print(token)
    string = tokenizer.decode(tokens['input_ids'])
    print(string)
    model = AutoModelWithLMHead.from_pretrained("dbmdz/german-gpt2")'''
    dataset_word = AutSLDataset(
        db_path = '/data/sl_datasets/autsl/autsl_hands_lmdb',
        sub_sample = False,
    )
    for i in range(len(dataset_word)):
        tens = dataset_word[i]['keypoint']
        if len(tens.shape) == 1:
            print(dataset_word[i]['index'], tens)
            continue
        a, b, c = tens.shape
        if a< 10:
            print(dataset_word[i]['index'], a)
    exit()
    save_list = []
    for i in range(len(dataset_word)):
        save_list.append((dataset_word[i]['index'], dataset_word[i]['text']))
        if i%100 == 0:
            print(i)
    save_list.sort()
    for i in save_list:
        print(i)
    with open('nia_label_list.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(f'{tup[0]} {tup[1]}' for tup in save_list[:3000]))
    renderer = KeypointRenderer()
    count = 0
    print(len(dataset_word))
    data_short = dataset_word
    count = 0

    #renderer.save_animation(reference = data2, generated = data_short[2]['input'])
    '''for data in data_short:
        print(data['input'])
        print(data['input'].shape)
        for dat in data['input']:
            x = dat[:,0]
            y = dat[:,1]
            plt.scatter(x,y)
            plt.savefig('foo.png')
            break
        break
    a = 0'''
    '''for data in dataset_word:
        a += 1
        if a % 1000 == 0:
            print(dataset_word.maxx)
            print(dataset_word.maxy)'''
    
    
    '''word_dict = defaultdict(int)
    i = 0

    for _, label in dataset_word:
        word_dict[label] += 1
        i+= 1
        if i % 500 == 0:
            print(i)
    
    json_word = json.dumps([word_dict])
    with open('NIA_classes1.json', "w") as f:
        f.write(json_word)'''


    
    """dataset_nia = NiaSLDataset(
        fpath = '/data/sl_datasets/niasl21/test/labeled_data/json_files',
        min_seq_len = 32,
        normalize = True
    )
    dataset = Phoenix2014TDataset(
        fpath = '/home/ejhwang/projects/phoenix14t/data/phoenix14t.pose.test',
        min_seq_len = 32,
        normalize = True
    )
    print(dataset_nia[100])
    print(dataset_nia[100]["joint_feats"].size())
    #print(dataset[0])
    #print(dataset[0]["joint_feats"].size())"""
    #import IPython; IPython.embed(); exit(1)    

    
    
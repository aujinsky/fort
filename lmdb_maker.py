import lmdb
import data_slr
import data
import torch
import torchtext
import pickle
import json
import os
import glob
import copy
from tqdm import tqdm
from slr import transformation
from torch.utils.data import DataLoader
import math
import random
weight = torch.tensor([[0.9809],
          [0.9857],
          [1.0774],
          [1.0033],
          [1.2062],
          [0.9713],
          [0.9402],
          [0.9230],
          [0.9780],
          [0.9590],
          [0.9915],
          [0.9706],
          [1.0069],
          [1.0160],
          [1.0476],
          [1.0328],
          [1.0552],
          [1.0674],
          [1.0071],
          [0.9994],
          [1.0131],
          [1.0538],
          [1.0278],
          [0.9933],
          [1.0015],
          [1.0143],
          [1.0331],
          [0.9499],
          [1.0288],
          [1.0218],
          [1.0217],
          [1.0062],
          [1.0160],
          [0.9832],
          [0.9931],
          [1.0083],
          [1.0068],
          [0.9905],
          [1.0036],
          [0.9920],
          [0.9638],
          [0.9652],
          [1.0270],
          [0.9682],
          [0.9619],
          [0.9620],
          [1.0046],
          [0.9908],
          [0.9494],
          [0.9563],
          [0.9727],
          [0.9646],
          [0.9640],
          [0.9709]])
weight_67 = torch.ones(67,1)
weight_67[25:46] = weight[12:33]
weight_67[46:67] = weight[33:54]
weight_67[0:8] = weight[0:8]
weight_67[15:19] = weight[8:12]
def rotate_helper_function(keypoints, angle, spec):
    # find center without normalizing the data
    joint_idx = 1
    
    batch_size, nframes, njoints, nfeats = keypoints.shape # batch_size is 1 though
    joint_coords = keypoints[:, :, joint_idx, :]

    if not isinstance(angle, dict):
        centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
        centered_keypoints_transformed = transformation.rotate(centered_keypoints, angle)
    else:
        import IPython; IPython.embed(); exit(1)
        RIGHT_WRIST_SKELETON = torch.tensor((0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
        LEFT_WRIST_SKELETON = torch.tensor((0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
        LOWER_HALF_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1), dtype=torch.float32)
        UPPER_HALF_SKELETON = torch.tensor((1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
        FACE_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0), dtype=torch.float32)
        #centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
        pose_centered_keypoints_transformed = transformation.rotate(keypoints, angle['POSE'])[:,:,:25]
        if 'rwrist' in spec:
            pose_centered_keypoints_transformed = keypoints
            pose_centered_keypoints_transformed[:,:,4] = transformation.rotate(keypoints, angle['POSE'])[:,:,4]
        if 'lwrist' in spec:
            pose_centered_keypoints_transformed 
        rhand_centered_keypoints_transformed = transformation.rotate(keypoints, angle['RHAND'])[:,:,25:46]
        lhand_centered_keypoints_transformed = transformation.rotate(keypoints, angle['LHAND'])[:,:,46:67]
        
        centered_keypoints_transformed = torch.cat((pose_centered_keypoints_transformed, rhand_centered_keypoints_transformed, lhand_centered_keypoints_transformed), dim=2)
        
    return centered_keypoints_transformed + joint_coords.view(batch_size, nframes, 1, nfeats)

def zoom_helper_function(keypoints, var, spec):
    # find center without normalizing the data
    joint_idx = 1
    
    batch_size, nframes, njoints, nfeats = keypoints.shape # batch_size is 1 though

    if not isinstance(var, dict):
        centered_keypoints_transformed = var * keypoints
    else:
        LOWER_HALF_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1), dtype=torch.float32)
        UPPER_HALF_SKELETON = torch.tensor((1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
        FACE_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0), dtype=torch.float32)
        #centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
        pose_centered_keypoints_transformed = (keypoints*var['POSE'])[:,:,:25]
        rhand_centered_keypoints_transformed = (keypoints*var['RHAND'])[:,:,25:46]
        lhand_centered_keypoints_transformed = (keypoints*var['LHAND'])[:,:,46:67]    
        centered_keypoints_transformed = torch.cat((pose_centered_keypoints_transformed, rhand_centered_keypoints_transformed, lhand_centered_keypoints_transformed), dim=2)
    
    return centered_keypoints_transformed

def flip_helper_function(keypoints, var):
    # find center without normalizing the data
    flipped_keypoints_transformed = torch.zeros(keypoints.shape)
    if var['POSE'] == 1.:
        flipped_keypoints_transformed[:,:,:25]  = transformation.flip(keypoints)[:,:,:25]
    else:
        flipped_keypoints_transformed[:,:,:25]  = keypoints[:,:,:25]
    if var['RHAND'] == 1.:
        flipped_keypoints_transformed[:,:,25:46] = transformation.flip(keypoints)[:,:,25:46]
    else:
        flipped_keypoints_transformed[:,:,25:46]  = keypoints[:,:,25:46]
    if var['LHAND'] == 1.:
        flipped_keypoints_transformed[:,:,46:67] = transformation.flip(keypoints)[:,:,46:67] 
    else:
        flipped_keypoints_transformed[:,:,46:67]  = keypoints[:,:,46:67]
    # 어짜피 centering을 afterwards에 하므로 상관 X.
    return flipped_keypoints_transformed

def gaussian_helper_function(keypoints, var, spec):
    # find center without normalizing the data
    if isinstance(var, float):
        noise = torch.zeros(keypoints.shape) + var
    elif 'weight' in var:
        noise = 1000.**(1.-weight_67) * 0.02
    else:
        if keypoints.shape[2] == 67:
            LOWER_HALF_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1), dtype=torch.float32)
            UPPER_HALF_SKELETON = torch.tensor((1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
            FACE_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0), dtype=torch.float32)
        elif keypoints.shape[2] == 54:
            LOWER_HALF_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,0,0,0,0), dtype=torch.float32)
            UPPER_HALF_SKELETON = torch.tensor((1,1,1,1,1,1,1,1,0,0,0,0), dtype=torch.float32)
            FACE_SKELETON = torch.tensor((0,0,0,0,0,0,0,0,1,1,1,1), dtype=torch.float32)
        pose_mask = spec[0] * LOWER_HALF_SKELETON + spec[1] * UPPER_HALF_SKELETON + spec[2] * FACE_SKELETON
        pose_var = pose_mask*var['POSE'] + (1-pose_mask)*var['RHAND']
        var = torch.cat((pose_var, torch.ones(21)*var['RHAND'], torch.ones(21)*var['LHAND']), dim=0).reshape(1, 1, keypoints.shape[2], 1)
        noise = torch.zeros(keypoints.shape) + var
    gaussian_keypoints_transformed = keypoints + torch.normal(0, noise)
    # 어짜피 centering을 afterwards에 하므로 상관 X.
    return gaussian_keypoints_transformed

# 8 9 10 11 12 13 14 19 20 21 22 23 24 

def arrange_sequence(n, m):
    if n == m:
        # If n is equal to m, return the sequence as it is.
        return torch.arange(0, n)
    elif n < m:
        index_to_use = torch.sort(torch.randperm(m)[:n])
        return index_to_use.values
    else: # n > m
        temp = torch.sort(torch.randperm(n-1)[:m-1])
        temp = temp.values +1
        end = torch.concat((temp, torch.tensor(n,).unsqueeze(0)))
        start = torch.concat((torch.tensor(0,).unsqueeze(0), temp))
        freq = end - start
        index_to_use_list = []
        for i in range(len(freq)):
            index_to_use_list = index_to_use_list + [i] * freq[i]
        return torch.tensor(index_to_use_list)


def velocity_helper_function(keypoints, scalar):
    return transformation.velocity_scalar(keypoints, scalar)

def fixed_size_n_frames_helper_function(keypoints, n=128):
    m=keypoints.shape[1]
    arranged_sequence = arrange_sequence(n, m)
    
    
    # Use the arranged sequence to index the tensor rows
    reordered_tensor = keypoints[:, arranged_sequence, :]
    print(reordered_tensor.shape)
    return reordered_tensor

def mainpulate_bone_helper_function(keypoints, scale=0.1):
    keypoints = keypoints.squeeze(0)
    keypoints = transformation.manipulate_bone(keypoints, scale)
    return keypoints.unsqueeze(0)


if __name__ == '__main__':
    number = 100

    augment_name = 'rotate_54_normalize_neo_hand_0_0.5_rwrist__8_0_0'


    asl = '/data/sl_datasets/wlasl/asl%d.json'%number

    if not os.path.isdir('/data/sl_datasets/wlasl/wlasl%d_lmdb_val_original'%number):
        with open(asl, 'r') as file:
            asl = json.load(file)
        splits = ["train", "val", "test"]
        for split in splits:
            print('/data/sl_datasets/wlasl/wlasl%d_lmdb_'%number+split+'_original')
            env = lmdb.open('/data/sl_datasets/wlasl/wlasl%d_lmdb_'%number+split+'_original', map_size=int(2e10))
            index = 0
            with env.begin(write=True) as txn:
                for p, instance in enumerate(tqdm(asl)):
                    for i in range(len(instance['instances'])):
                        item = {}
                        item["text"] = instance['gloss']
                        item["gloss"] = instance['gloss']
                        video_id = instance['instances'][i]['video_id']
                        item["index"] = video_id
                        frame_start = int(instance['instances'][i]['frame_start'])
                        frame_end = int(instance['instances'][i]['frame_end'])
                        decider = instance['instances'][i]['split']
                        if not split == decider:
                            continue
                        skel_list = []
                        for j in range(frame_start, frame_end+1):
                            if not os.path.isfile("/data/sl_datasets/wlasl/pose_per_individual_videos/"+video_id+("/image_%05d_keypoints.json"%j)):
                                continue
                            with open("/data/sl_datasets/wlasl/pose_per_individual_videos/"+video_id+("/image_%05d_keypoints.json"%j), 'r') as individual_frame_file:
                                try:
                                    individual_frame = json.load(individual_frame_file)
                                    pose = individual_frame['people'][0]['pose_keypoints_2d']
                                    left_hand = individual_frame['people'][0]['hand_left_keypoints_2d']
                                    right_hand = individual_frame['people'][0]['hand_right_keypoints_2d']
                                    del pose[2::3]
                                    del right_hand[2::3]
                                    del left_hand[2::3]
                                    skel_list += [pose + right_hand + left_hand]
                                except:
                                    if individual_frame['people'] == []:
                                        continue
                                    import IPython; IPython.embed(); exit(1)

                        skel_tensor = torch.Tensor(skel_list)
                        item["keypoint"] = skel_tensor
                        item["dataset_type"] = "org" 
                        lmdb_item = pickle.dumps(item)
                        txn.put(str(index).encode("utf-8"), lmdb_item)
                        index = index + 1
                print(index)
        import IPython; IPython.embed(); exit(1)
    #raise NotImplementedError

    env = lmdb.open(('/data/sl_datasets/wlasl/wlasl%d_lmdb_train_'%number) +augment_name, map_size=int(2e10))
    #env = lmdb.open(('/data/sl_datasets/niasl20/nia20_lmdb_train_') +augment_name, map_size=int(2e10))

    if 'original' in augment_name:
        normalize = False
        normalize_hand = False

    if 'train_normalize' in augment_name or 'val_normalize' in augment_name or 'test_normalize' in augment_name:
        normalize = True
        if 'hand' in augment_name:
            normalize_hand = True
        else:
            normalize_hand = False
    elif 'normalize' in augment_name:
        normalize = False
        normalize_hand = False
    if 'hand' in augment_name:
        db_path = '/data/sl_datasets/wlasl/wlasl%d_lmdb_train_normalize_neo_hand_0_0.5'%number
    elif 'hand' not in augment_name:
        db_path = '/data/sl_datasets/wlasl/wlasl%d_lmdb_train_normalize_0.5'%number
    print("db_path: ", db_path)
    dataset_cmp = data.WlaSLDataset(
        db_path = db_path, seq_len = -1, min_seq_len = -1, normalize = normalize, padding = False, sub_sample = False, normalize_hand = normalize_hand
    )
    """
    dataset_cmp = data.WlaSLDataset(
        db_path = '/data/sl_datasets/niasl20/nia20_lmdb_train_full', seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    """
    gloss_list = []
    if not os.path.isfile("/home/ajkim/kslt/slr/WLASL%d_GLOSSES.json"%number):
        for elem in tqdm(dataset_cmp):
            if not elem["gloss"] == "":
                gloss_list.append([elem["gloss"]])
        gloss_vocab = torchtext.vocab.build_vocab_from_iterator(gloss_list, specials=["<unk>"])
        vocab_json = json.dumps(gloss_vocab.get_stoi())
        with open("/home/ajkim/kslt/slr/WLASL%d_GLOSSES.json"%number, 'w') as f:
            f.write(vocab_json)
        print(len(gloss_vocab))
        import IPython; IPython.embed(); exit(1)

    random.seed(42)
    torch.manual_seed(42)

    origin_dataloader = DataLoader(dataset_cmp, batch_size=1, shuffle=False)
    var = None
    print(augment_name)

    i =0
    with env.begin(write=True) as txn:

        if '__' in augment_name:
            var_string = augment_name.split('__')[1]
            var_list = var_string.split("_")
            if len(var_list) == 1 and not 'weight' in var_list[0]:
                var = float(var_list[0])
            elif len(var_list) == 1 and 'weight' in var_list[0]:
                var = var_list[0]
            else:
                var = {'POSE': float(var_list[0]), 'RHAND': float(var_list[1]), 'LHAND': float(var_list[2])}
            spec = [True, True, True]
            if "lower" in augment_name or "upper" in augment_name or "face" in augment_name:
                spec = ["lower" in augment_name, "upper" in augment_name, "face" in augment_name]
            if "wrist" in augment_name or "elbow" in augment_name or "shoulder" in augment_name:
                spec = "wrist"
        for index, data in tqdm(enumerate(origin_dataloader)):
            """
            if index == 1442:
                import IPython; IPython.embed(); exit(1)
            """
            item = {}
            item["text"] = data["text"]
            item["gloss"] = data["gloss"]
            item["keypoint"] = data["keypoint"]
            item["index"] = data["index"]

            if not isinstance(data["text"], str):
                item["text"] = item["text"][0]
                item["gloss"] = item["gloss"][0]
                item["index"] = item["index"][0]

            #item["keypoint"] = mainpulate_bone_helper_function(item["keypoint"], 0.01)
            item["dataset_type"] = augment_name[:3] # res # fli # rot # gau # def
            if augment_name[:3] == "rot" and not('first' in augment_name):
                if isinstance(var, dict):
                    k = {'POSE': random.uniform(-var['POSE'], var['POSE'])*math.pi/180, 'RHAND': random.uniform(-var['RHAND'], var['RHAND'])*math.pi/180, 'LHAND': random.uniform(-var['LHAND'], var['LHAND'])*math.pi/180}
                else:
                    k = random.uniform(-13, 13)*math.pi/180
                item["keypoint"] = rotate_helper_function(item["keypoint"], k, spec).squeeze(0)
            elif augment_name[:3] == "fli":
                item["keypoint"] = flip_helper_function(item["keypoint"], var).squeeze(0)
            elif augment_name[:3] == "gau":
                item["keypoint"] = gaussian_helper_function(item["keypoint"], var, spec).squeeze(0)
            elif augment_name[:3] == "res":
                item["keypoint"] = rotate_helper_function(item["keypoint"], 0.02).squeeze(0)
            elif augment_name[:3] == "bon":
                item["keypoint"] = mainpulate_bone_helper_function(item["keypoint"], scale=0.03).squeeze(0)
            elif augment_name[:3] == "zoo":
                if isinstance(var, dict):
                    k = {'POSE': random.uniform(1-var['POSE'], 1.), 'RHAND': random.uniform(1-var['RHAND'], 1.), 'LHAND': random.uniform(1-var['LHAND'], 1.)}
                else:
                    k = random.uniform(1-var, 1.)
                item["keypoint"] = zoom_helper_function(item["keypoint"], k, spec).squeeze(0)
            elif augment_name[:3] == "2x_":
                item["seq_index"] = index
                item["keypoint"] = item["keypoint"].squeeze(0)
                item2 = copy.deepcopy(item)
                item2["seq_index"] = index + 1442
            else:
                if not augment_name[-2:] == "54":
                    item["keypoint"] = item["keypoint"].squeeze(0)
                elif augment_name[-2:] == "54" and (not item["keypoint"].squeeze(0).shape[1] == 137):
                    item["keypoint"] = item["keypoint"].squeeze(0)
                    body12_index = torch.cat((torch.tensor([0,1,2,3,4,5,6,7,15,16,17,18]), torch.arange(25, 67)))
                    item["keypoint"] = item["keypoint"].index_select(1, body12_index)
                    # import IPython; IPython.embed(); exit(1)
                elif augment_name[-2:] == "54" and item["keypoint"].squeeze(0).shape[1] == 137:
                    item["keypoint"] = item["keypoint"].squeeze(0)
                    right = item["keypoint"][:,95:116]
                    left = item["keypoint"][:,116:]
                    body = item["keypoint"][:,70:89]
                    face = item["keypoint"][:,0:70]
                    item["keypoint"] = torch.cat((body, right, left), dim=1)
                    body12_index = torch.cat((torch.tensor([0,1,2,3,4,5,6,7,15,16,17,18]), torch.arange(19, 61)))
                    item["keypoint"] = item["keypoint"].index_select(1, body12_index)
            
            lmdb_item = pickle.dumps(item)
            txn.put(str(index).encode("utf-8"), lmdb_item)
            if augment_name[:3] == "2x_":
                lmdb_item2 = pickle.dumps(item2)
                txn.put(str(index+1442).encode("utf-8"), lmdb_item2)
            
    if not (var==None):
        print(item["keypoint"].shape)
        #print(k)
        print(var, spec, item["keypoint"][0,0,0], item["keypoint"][0, 40, 0])
        print(var, spec, item["keypoint"][0,0,1], item["keypoint"][0, 40, 1])
        print('normalized_right_before: ', normalize)

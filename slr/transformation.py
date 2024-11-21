import torch
import math
from einops_exts import check_shape
from .graph import *

def scale(keypoints_tensor, center=0., coeff=0.8, static=False):
    return (keypoints_tensor - center) * ((1-coeff) * torch.rand(1).to(keypoints_tensor.device) + coeff)

def rotate(keypoints_tensor, center=0., coeff=0.5, static=False):
    angle = (torch.rand(1)*2 -1)*coeff
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.matmul((keypoints_tensor - center) , torch.tensor(((cos, -sin), (sin, cos))).to(keypoints_tensor.device))

def flip(keypoints_tensor):
    return torch.matmul(keypoints_tensor, torch.tensor(((1., 0.),(0., -1.))).to(keypoints_tensor.device))

def masking(keypoints_tensor):
    return

def velocity_scalar(keypoints_tensor, scalar):
    if scalar > 1.:
        return keypoints_tensor[0::int(scalar)]
    elif scalar < 1.:
        NotImplementedError # not wise to increase more memomry 
    else:
        return keypoints_tensor
    
def manipulate_single_bone(keypoints_tensor, bone_type, num_of_index_of_bone, new_scale): # same goes to all the frames
    # input shape frames, 137, 2
    if keypoints_tensor.shape[1] == 54:
        HAND_START_INDEX = 12
    elif keypoints_tensor.shape[1] == 67:
        HAND_START_INDEX = 25 
    HAND_NUM_KEYPOINTS = 21
    if bone_type == 'POSE': 
        ini, fin = POSE_MAP[num_of_index_of_bone]
        bone_vector = keypoints_tensor[:, fin] - keypoints_tensor[:, ini]
        new_bone_vector = new_scale * bone_vector
        diff = (new_bone_vector - bone_vector).unsqueeze(1) # frames 2
        mask = POSE_CHILD_MATRIX[fin].unsqueeze(0).unsqueeze(2)
        keypoints_tensor[:, :HAND_START_INDEX ] = keypoints_tensor[:, :HAND_START_INDEX ] + diff * mask
        if POSE_CHILD_MATRIX[fin, 4] == 1:
            keypoints_tensor[:, HAND_START_INDEX :HAND_START_INDEX + HAND_NUM_KEYPOINTS] = keypoints_tensor[:, HAND_START_INDEX :HAND_START_INDEX + HAND_NUM_KEYPOINTS] + diff
        if POSE_CHILD_MATRIX[fin, 7] == 1:
            keypoints_tensor[:, HAND_START_INDEX + HAND_NUM_KEYPOINTS:HAND_START_INDEX + HAND_NUM_KEYPOINTS * 2] = keypoints_tensor[:, HAND_START_INDEX + HAND_NUM_KEYPOINTS:HAND_START_INDEX + HAND_NUM_KEYPOINTS * 2] + diff
    if bone_type == 'RHAND':
        ini, fin = HAND_MAP[num_of_index_of_bone]
        bone_vector = keypoints_tensor[:,fin] - keypoints_tensor[:,ini]
        new_bone_vector = new_scale * bone_vector
        diff = (new_bone_vector - bone_vector).unsqueeze(1)
        mask = HAND_CHILD_MATRIX[fin].unsqueeze(0).unsqueeze(2)
        keypoints_tensor[:, HAND_START_INDEX :HAND_START_INDEX + HAND_NUM_KEYPOINTS] = keypoints_tensor[:, HAND_START_INDEX :HAND_START_INDEX + HAND_NUM_KEYPOINTS] + diff * mask
    if bone_type == 'LHAND':
        ini, fin = HAND_MAP[num_of_index_of_bone]
        bone_vector = keypoints_tensor[:,fin] - keypoints_tensor[:,ini]
        new_bone_vector = new_scale * bone_vector
        diff = (new_bone_vector - bone_vector).unsqueeze(1)
        mask = HAND_CHILD_MATRIX[fin].unsqueeze(0).unsqueeze(2)
        keypoints_tensor[:, HAND_START_INDEX + HAND_NUM_KEYPOINTS:HAND_START_INDEX + HAND_NUM_KEYPOINTS * 2] = keypoints_tensor[:, HAND_START_INDEX + HAND_NUM_KEYPOINTS:HAND_START_INDEX + HAND_NUM_KEYPOINTS * 2] + diff * mask
    return keypoints_tensor

def manipulate_bone(keypoints_tensor, scalar, specific = {'POSE': True, 'RHAND': True, 'LHAND': True}):
    pose_mean = torch.ones(len(POSE_MAP), 2)
    pose_std = scalar * torch.ones(len(POSE_MAP), 2)
    POSE_MANIP_VAL = torch.normal(pose_mean, pose_std)
    rhand_mean = torch.ones(len(HAND_MAP), 2)
    rhand_std = scalar * torch.ones(len(HAND_MAP), 2)
    RHAND_MANIP_VAL = torch.normal(rhand_mean, rhand_std)
    lhand_mean = torch.ones(len(HAND_MAP), 2)
    lhand_std = scalar * torch.ones(len(HAND_MAP), 2)
    LHAND_MANIP_VAL = torch.normal(lhand_mean, lhand_std)
    modified_keypoints_tensor = keypoints_tensor
    # order skel_list += [pose + right_hand + left_hand + landmark]
    if specific['POSE']:
        for i in range(len(POSE_MAP)):
            r = modified_keypoints_tensor
            modified_keypoints_tensor = manipulate_single_bone(modified_keypoints_tensor, 'POSE', i, POSE_MANIP_VAL[i])
    if specific['RHAND']:
        for i in range(len(HAND_MAP)):
            modified_keypoints_tensor = manipulate_single_bone(modified_keypoints_tensor, 'RHAND', i, RHAND_MANIP_VAL[i])
    if specific['LHAND']:
        for i in range(len(HAND_MAP)):
            modified_keypoints_tensor = manipulate_single_bone(modified_keypoints_tensor, 'LHAND', i, LHAND_MANIP_VAL[i])
    return modified_keypoints_tensor

def preprocess_keypoints(keypoints_tensor, norm = None, remove_noisy = True, normalize_hand = False, **kwargs):
    #print(keypoints_tensor.shape)
    def normalize_skeleton(keypoints, nfeats):
        if nfeats == 3:
            return normalize_skeleton_3d(keypoints)
        elif nfeats == 2:
            return normalize_skeleton_2d(keypoints)
        else:
            raise ValueError(f"Invalid number of features: {nfeats}")
    
    check_shape(keypoints_tensor, "f v c")
    nfeats = keypoints_tensor.shape[-1]
    
    if remove_noisy == True:
        keypoints_tensor = remove_noisy_frames(keypoints_tensor, **kwargs)
    keypoints_tensor_original = keypoints_tensor
    
    #keypoints_tensor = normalize_skeleton(keypoints_tensor, nfeats)
    
    if norm == 'z':
        keypoints_tensor = z_normalize(keypoints_tensor)
    elif norm == 'norm':
        if not keypoints_tensor.shape[0] == 0:
            keypoints_tensor = normalize(keypoints_tensor, normalize_hand, keypoints_tensor_original)

    return keypoints_tensor


def remove_noisy_frames(X, threshold=200.):
    """
    Remove noisy frames based on the Euclidean distance between consecutive frames.
    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints
        threshold: threshold value for the Euclidean distance, frames with distance above the threshold are removed
    Returns:
        X_clean: torch tensor of shape (num_clean_frames, num_joints, 3) representing the cleaned 3D keypoints
    """
    num_frames, num_joints, _ = X.shape
    X_diff = torch.diff(X, dim=0)  # compute the difference between consecutive frames
    distances = torch.norm(X_diff, dim=-1)  # compute the Euclidean distance
    distances = torch.mean(distances, dim=-1)

    mask = torch.ones(num_frames, dtype=torch.bool)  # initialize a mask to keep all frames
    
    mask[1:] = distances <= threshold  # set to False all frames with distance above the threshold
    X_clean = X[mask]  # apply the mask to the input keypoints
    
    return X_clean


def normalize_skeleton_3d(X, resize_factor=None):
    def distance_3d(x1, y1, z1, x2, y2, z2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

    anchor_pt = X[:, 1, :].reshape(-1, 3)  # neck

    if resize_factor is None:
        neck_height = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2],
                                  X[:, 0, 0], X[:, 0, 1], X[:, 0, 2]).float()
        shoulder_length = distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 2, 0], X[:, 2, 1], X[:, 2, 2]) + \
                          distance_3d(X[:, 1, 0], X[:, 1, 1], X[:, 1, 2], X[:, 5, 0], X[:, 5, 1], X[:, 5, 2])
        resized_neck_height = (neck_height / shoulder_length).mean()
        if resized_neck_height > 0.6:
            resize_factor = shoulder_length * resized_neck_height / 0.6
        else:
            resize_factor = shoulder_length

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor
        normalized_X[:, i, 2] = (X[:, i, 2] - anchor_pt[:, 2]) / resize_factor

    return normalized_X


def center_keypoints(keypoints, joint_idx=1):
    """
    Center the keypoints around a specific joint.
    
    Args:
        keypoints (torch.Tensor): Tensor of shape (batch_size, nframes, njoints, nfeat) or (nframes, njoints, nfeat) containing the keypoints.
        joint_idx (int): Index of the joint to center the keypoints around.
        
    Returns:
        torch.Tensor: Tensor of the same shape as input with the keypoints centered around the specified joint.
    """
    if len(keypoints.shape) == 4:
        batch_size, nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, :, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(batch_size, nframes, 1, nfeats)
    elif len(keypoints.shape) == 3:
        nframes, njoints, nfeats = keypoints.shape
        joint_coords = keypoints[:, joint_idx, :]
        centered_keypoints = keypoints - joint_coords.view(nframes, 1, nfeats)
    else:
        raise ValueError("Input keypoints tensor must have either 3 or 4 dimensions")

    return centered_keypoints


def normalize(X, normalize_hand=False, original_X = None):
    """
    Normalize 3D keypoints using min-max normalization.
    Args:
        X: torch tensor of shape (num_frames, num_joints, 3) representing 3D keypoints
    Returns:
        X_norm: torch tensor of shape (num_frames, num_joints, 3) representing the normalized 3D keypoints
    """

    T, n, d = X.shape

    if n == 54:
        HAND_START_INDEX = 12
    elif n == 67:
        HAND_START_INDEX = 25 
    HAND_NUM_KEYPOINTS = 21

    X = X.reshape(T*n, d)
    X_min = torch.min(X, dim=0)[0] # min-max whole video wise
    X_max = torch.max(X, dim=0)[0]
    X_norm = -0.5 + 1.0 * (X - X_min) / (X_max - X_min)
    X_norm = X_norm.reshape(T, n, d)
    if normalize_hand == True:
        res = normalize_hand_skeleton(X.reshape(T, n, d), original_X)
        X_norm[:, HAND_START_INDEX :HAND_START_INDEX +HAND_NUM_KEYPOINTS*2] = res
    return X_norm

def z_normalize(X):
    X_x = X[:, :, 0]
    X_y = X[:, :, 1]
    std_x = torch.std(X_x)
    std_y = torch.std(X_y)
    return X / torch.cat((torch.Tensor([std_x]),torch.Tensor([std_y])))



def normalize_skeleton_2d(X, resize_factor=None):
    def distance_2d(x1, y1, x2, y2):
        return torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    # print(X.shape) length HAND_START_INDEX + HAND_NUM_KEYPOINTS * 2 2
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
           
    resize_factor = restore_list(resize_factor)

    normalized_X = X.clone()
    for i in range(X.shape[1]):
        normalized_X[:, i, 0] = (X[:, i, 0] - anchor_pt[:, 0]) / resize_factor
        normalized_X[:, i, 1] = (X[:, i, 1] - anchor_pt[:, 1]) / resize_factor
    
    return normalized_X


def normalize_hand_skeleton(keypoints, original_X):
    if keypoints.shape[1] == 54:
        HAND_START_INDEX = 12
    elif keypoints.shape[1] == 67:
        HAND_START_INDEX = 25 
    HAND_NUM_KEYPOINTS = 21
    HAND_RIGHT = 0 
    HAND_LEFT = 1
    normalized_hands = []
    for hand_index in range(2):
        # Treat each element of the sequence (analyzed frame) individually
        hand_start = HAND_START_INDEX+hand_index*HAND_NUM_KEYPOINTS
        hand_end = HAND_START_INDEX+(hand_index+1)*HAND_NUM_KEYPOINTS
            
        landmarks_x_values = keypoints[:, hand_start:hand_end, 0]
        landmarks_y_values = keypoints[:, hand_start:hand_end, 1]
        
        # Calculate the deltas
        width, height = torch.max(landmarks_x_values, dim=1).values - torch.min(landmarks_x_values, dim=1).values, torch.max(landmarks_y_values, dim=1).values - torch.min(landmarks_y_values, dim=1).values
        mask = width > height
        delta_x = (0.1 * width) * mask
        delta_y = (delta_x + ((width - height) / 2)) * mask
        delta_y = delta_y + 0.1 * height * (~mask)
        delta_x = delta_x + (delta_y + ((height - width) / 2)) * (~mask)
        

        # Set the starting and ending point of the normalization bounding box
        starting_point = (torch.min(landmarks_x_values, dim=1).values - delta_x, torch.min(landmarks_y_values, dim=1).values - delta_y)
        ending_point = (torch.max(landmarks_x_values, dim=1).values + delta_x, torch.max(landmarks_y_values, dim=1).values + delta_y)

        
        zero_mask = ((ending_point[0] - starting_point[0]) == 0) | ((starting_point[1] - ending_point[1]) == 0) | (original_X[:, hand_start:hand_end, :]==0).any(dim=2).any(dim=1)
        
        if zero_mask.any():
            nonabnormal = -1
            abnormal = zero_mask.nonzero()
            
            if len(((zero_mask == 0).nonzero())) == 0:
                normalized_x = torch.zeros(keypoints.shape[0],HAND_NUM_KEYPOINTS)
                normalized_y = torch.zeros(keypoints.shape[0],HAND_NUM_KEYPOINTS)
                normalized_hands.append(torch.stack((normalized_x, normalized_y), dim=2))
                continue
            for i in range(abnormal.shape[0]):
                if abnormal[i, 0] > nonabnormal:
                    nonabnormal = abnormal[i, 0] - 1
                    if nonabnormal == -1:
                        first_nonabnormal = ((zero_mask == 0).nonzero())[0, 0] # first nonabnormal frame
                        landmarks_x_values[abnormal[i, 0], :] = landmarks_x_values[first_nonabnormal, :]
                        starting_point[0][abnormal[i, 0]] = starting_point[0][first_nonabnormal]
                        ending_point[0][abnormal[i, 0]] = ending_point[0][first_nonabnormal]
                        landmarks_y_values[abnormal[i, 0], :] = landmarks_y_values[first_nonabnormal, :]
                        starting_point[1][abnormal[i, 0]] = starting_point[1][first_nonabnormal]
                        ending_point[1][abnormal[i, 0]] = ending_point[1][first_nonabnormal]
                        nonabnormal = 0
                    else:
                        landmarks_x_values[abnormal[i, 0], :] = landmarks_x_values[nonabnormal, :]
                        starting_point[0][abnormal[i, 0]] = starting_point[0][nonabnormal]
                        ending_point[0][abnormal[i, 0]] = ending_point[0][nonabnormal]
                        landmarks_y_values[abnormal[i, 0], :] = landmarks_y_values[nonabnormal, :]
                        starting_point[1][abnormal[i, 0]] = starting_point[1][nonabnormal]
                        ending_point[1][abnormal[i, 0]] = ending_point[1][nonabnormal]

        
        normalized_x = (landmarks_x_values - starting_point[0].unsqueeze(1)) / (ending_point[0] - starting_point[0]).unsqueeze(1) - 0.5
        normalized_y = (landmarks_y_values - starting_point[1].unsqueeze(1)) / (ending_point[1] - starting_point[1]).unsqueeze(1) - 0.5
        normalized_hands.append(torch.stack((normalized_x, normalized_y), dim=2))
    
    return torch.cat((normalized_hands[0], normalized_hands[1]), dim = 1)

def scaling_keypoints(keypoints, width=800, height=900, is_2d=False):
    """
    Unnormalize keypoints by scaling the coordinates based on the width and height
    of the original image.
    Args:
        keypoints: a PyTorch tensor of shape (batch_size, num_frames, num_joints, coord_dim) or (num_frames, num_joints, coord_dim)
            representing keypoints
        width: width of the original image
        height: height of the original image
        is_2d: bool, if True, assumes keypoints are 2D, otherwise assumes keypoints are 3D
    Returns:
        unnormalized_keypoints: a PyTorch tensor of the same shape as keypoints, with each element
        unnormalized based on the width, height, and depth (if 3D) of the original image
    """
    coord_dim = keypoints.shape[-1]
    if coord_dim == 2:
        is_2d = True
    else:
        is_2d = False

    # Scale the x and y coordinates based on the width and height of the original image
    unnormalized_keypoints = keypoints.clone()
    unnormalized_keypoints[..., 0] *= width
    unnormalized_keypoints[..., 1] *= height

    if not is_2d:
        depth = (width + height) * 0.5
        unnormalized_keypoints[..., 2] *= depth

    return unnormalized_keypoints

def restore_list(input):
    l = len(input)
    if 0 not in input:
        return input
    
    for i in range(l):
        if i == 0 and input[i] == 0:
            for r in range(1, l):
                if not input[r] == 0:
                    input[i] = input[r]
                    break
        elif (not i == 0) and input[i] == 0:
            j = 1
            while i+j < l and input[i+j] == 0:
                j += 1
            after = 0 if i+j == l else input[i+j]
            before = 0 if i == 0 else input[i-1]
            replace = after + before
            if after !=0 and before != 0:
                replace /= 2
            for k in range(j):
                input[i+k] = replace
    return input    
            

def restore_list2(input): # if problem occurs, change to restore_list2
    l = len(input)
    if 0 not in input:
        return input
    
    for i in range(l):
        if input[i] == 0:
            j = 1
            while i+j < l and input[i+j] != 0:
                j += 1
            after = 0 if i+j == l else input[i+j]
            before = 0 if i == 0 else input[i-1]
            replace = after + before
            if after !=0 and before != 0:
                replace /= 2
            for k in range(j):
                input[i+k] = replace
    return input    
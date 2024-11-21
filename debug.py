import data_torch
from tqdm import tqdm
import data
import data_module
from torch.utils.data import DataLoader, Subset

if True:
    def check_nan(l):
        for i in tqdm(range(len(l))):
            if l[i]['joint_feats'].isnan().any():
                print(i, l[i])
                return l.isnan().any()

    def check_zero(l, joint_idx):
        count = 1
        for i in tqdm(range(len(l))):
            if (l[i]['keypoint'][:,joint_idx,:] == 0).any():
                print(i, l[i]['keypoint'][:,joint_idx,:], count)
                count = count + 1

    def check_zero(item):
        print(item.shape[1])
        for i in range(item.shape[1]):
            if (item[:,i,:]==0).any():
                print(i)

    def check_module_nan(k):
            for i in tqdm(k):
                if i[0].isnan().any():
                    print(i)
                    return i.isnan().any()

if __name__ == '__main__':
    datamodule = data_module.SignDataModule("2D", dataset = 'phoenix', gloss = True, shuffle = False)
    datamodule.generate_gloss_vocab()
    #k = DataLoader(datamodule.trainset, batch_size=1, collate_fn=datamodule.train_collate_fn, num_workers=0, shuffle = False)
    #l = DataLoader(datamodule.trainset2, batch_size=32, collate_fn=datamodule.trainset2.collate_fn, num_workers=0, shuffle = False)

    """
    q = data.NiaSLDataset20(
        db_path = '/data/sl_datasets/niasl20/nia20_lmdb_train_gaussian_54_normalize_neo_hand_0_0.5_0.030.01_0.01'   
    )
    """

    a = data.AutSLDataset(
            db_path = '/data/sl_datasets/autsl/autsl_hands_lmdb',
            sub_sample = False,
        )

    r = data.WlaSLDataset(
        db_path = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_original', seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )

    k = data.WlaSLDataset(
        db_path = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_neo_54_hand_0_0.5', seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    x2 = data.WlaSLDataset(
        db_path = '/data/sl_datasets/wlasl/wlasl100_lmdb_train_2x_normalize_neo_54_hand_0_0.5', seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    l = data.WlaSLDataset(
        db_path = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_gaussian_54_normalize_neo_hand_0_0.5__0.01_0.01_0.1", seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    w5 = data.WlaSLDataset(
        db_path = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_0.5", seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    
    import IPython; IPython.embed(); exit(1)
else:
    ori = data.WlaSLDataset(
        db_path = '/data/sl_datasets/wlasl/wlasl100_lmdb_test_original', seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    n5 = data.WlaSLDataset(
        db_path = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_0.5", seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
    h5 = data.WlaSLDataset(
        db_path = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_neo_54_hand_0_0.5", seq_len = -1, min_seq_len = -1, normalize = False, padding = False, sub_sample = False, normalize_hand = False
    )
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchmetrics
from model_baseline_slr import SLRModel
from model_baseline_slr_weight import SLRWeightModel
from model_baseline_slr_weight_augment import SLRWeightAugmentModel
from model_baseline_slr_encoderdecoder import SLREncoderDecoderModel
from model_baseline_slr_encoderdecoder_weight import SLREncoderDecoderWeightModel
from model_baseline_slr_spoter import SLRSPOTERModel
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

temp_dict = {}



def gen_src_mask(total_len, len_list):
    batch_len = len(len_list)
    zero = torch.zeros(batch_len, total_len)
    for tens, t in zip(zero, len_list):
        mask = torch.ones(total_len-t) 
        tens[t:] = mask
    ret = zero.bool()
    return ret

def save_args(args, filename):
    args_dict = vars(args)
    with open(filename, 'w') as file:
        json.dump(args_dict, file, indent=4)


class TrainerModule(pl.LightningModule):
    def __init__(self, classes, dataset="niasl", **kwargs):
        super().__init__()
        self.save_hyperparameters({'dataset': dataset, 'kwargs': kwargs})
        #self.save_hyperparameters(kwargs)
        num_classes = len(classes)
        self.classes = classes
        self.dataset = dataset
        self.train_loss_list = []
        self.test_acc_list = []
        self.alpha = 0.0001
        print("alpha: %f" %(self.alpha))
        if self.dataset == "niasl":
            if not kwargs['body12']:
                dim = 137
            elif kwargs['body12']:
                dim = 54
        elif self.dataset == "wlasl":
            if not kwargs['body12']:
                dim = 67
            elif kwargs['body12']:
                dim = 54
        print(kwargs['body12'], dim)
        print(kwargs['model'])
        if kwargs['model'] == 'encoder':
            self.model = SLRModel(num_classes, dim, 2, 512, 6, mask = True)
        if kwargs['model'] == 'weight':
            self.model = SLRWeightModel(num_classes, dim, 2, 512, 6, mask = True)
        if kwargs['model'] == 'weight_augment':
            self.model = SLRWeightAugmentModel(num_classes, dim, 2, 512, 6, mask = True)
        if kwargs['model'] == 'encdec':
            self.model = SLREncoderDecoderModel(num_classes, dim, 8, 512, 6, mask = True)
        if kwargs['model'] == 'spoter':
            self.model = SLRSPOTERModel(num_classes, dim, 8, 512, 6, mask = True)
        if kwargs['model'] == 'spoter_weight':
            self.model = SLRSPOTERWeightModel(num_classes, dim, 2, 512, 6, mask = True)
        if kwargs['model'] == 'encdec_weight':
            self.model = SLREncoderDecoderWeightModel(num_classes, dim, 2, 512, 6, mask = True)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        if not kwargs['aug_method'] == []:
            aug_method = kwargs['aug_method'][0][0]
            if aug_method == 'rotate':
                self.aug_function = transformation.rotate
            elif aug_method == 'flip':
                self.aug_function = transformation.flip
            elif aug_method == 'scale':
                self.aug_function = transformation.scale
            elif aug_method == 'bone':
                self.aug_function = None
            elif aug_method == None:
                self.aug_function = None
    def remove_nan(self, l, z, labels):
        nan_samples = (torch.tensor(l)==0)
        z_remove_nan = z[~nan_samples,:]
        labels_remove_nan = labels[~nan_samples]
        return z_remove_nan, labels_remove_nan
    def training_step(self, batch, batch_idx):
        x = batch['keypoint']
        y = batch['gloss']
        l = batch['len_list']
        _, total_len, _, _ = x.shape
        mask = gen_src_mask(total_len,l).to('cuda')
        #mask = None
        temp_list = []
        
        for label in y:
            if type(label) == list:
                label = label[0]
            if label in self.classes:
                k = self.classes.index(label)
            else:
                k = -1 
                for i, subclass in enumerate(self.classes):
                    if "/" in subclass:
                        if set(label) <= set(subclass):
                            k = i
                            break
                if k == -1:
                    if "결심3" in label:
                        label = "결심4"
                    if "번째" in label:
                        if "여덟" in label:
                            label = "8번째"
                        if "열" in label:
                            label = "10번째"
                        if "열두" in label:
                            label = "12번째"
                        if "아홉" in label:
                            label = "9번째"
                    k = self.classes.index(label)
            temp_list.append(k)
        labels = torch.tensor(temp_list).to("cuda")
        if "index_list" in batch:
            z = self.model(x, mask, batch['index_list'])
        else:
            z = self.model(x, mask)
        z, labels = self.remove_nan(l, z, labels)
        loss = F.cross_entropy(z, labels)
        loss_tot = loss
        if hasattr(self.model, 'scale'):
            loss_tot = loss_tot + self.alpha * (0-torch.log(self.model.scale))
        if loss.isnan().any() == True:
            import IPython; IPython.embed(); exit(1)    
        self.log('train_loss_step', loss_tot)
        self.train_loss_list.append(loss_tot)
        return loss_tot
    def on_train_epoch_end(self):
        train_loss_tensor = torch.tensor(self.train_loss_list)
        self.train_loss_list = []
        average_train_loss = train_loss_tensor.mean()
        if hasattr(self.model, 'scale'):
            self.log('scale', self.model.scale)
        self.log('avg_train_loss', average_train_loss)

    def validation_step(self, batch, batch_idx):
        x = batch['keypoint']
        y = batch['gloss']
        l = batch['len_list']
        
        _, total_len, _, _ = x.shape
        
        mask = gen_src_mask(total_len,l).to('cuda')
        
        # [141, 108, 86, ..., 149]
        
        temp_list = []
        """
        for label in y:
            if type(label) == list:
                label = label[0]
            k = self.classes.index(label)
            temp_list.append(k)
        """
        for label in y:
            if type(label) == list:
                label = label[0]
            if label in self.classes:
                k = self.classes.index(label)
            else:
                k = -1 
                for i, subclass in enumerate(self.classes):
                    if "/" in subclass:
                        if set(label) <= set(subclass):
                            k = i
                            break
                if k == -1:
                    if "결심3" in label:
                        label = "결심4"
                    if "번째" in label:
                        if "여덟" in label:
                            label = "8번째"
                        if "열" in label:
                            label = "10번째"
                        if "열두" in label:
                            label = "12번째"
                        if "아홉" in label:
                            label = "9번째"
                    k = self.classes.index(label)
            temp_list.append(k)
        labels = torch.tensor(temp_list).to("cuda")
        z = self.model(x, mask)
        z, labels = self.remove_nan(l, z, labels)
        z_tot = z
        
        acc = self.accuracy(z_tot, labels)
        self.log('val_acc_step', acc)

    def test_step(self, batch, batch_idx):
        x = batch['keypoint']
        y = batch['gloss']
        l = batch['len_list']
        
        b, total_len, _, _ = x.shape
        
        mask = gen_src_mask(total_len,l).to('cuda')
        
        # [141, 108, 86, ..., 149]
        
        temp_list = []
        for label in y:
            if type(label) == list:
                label = label[0]
            if label in self.classes:
                k = self.classes.index(label)
            else:
                k = -1 
                for i, subclass in enumerate(self.classes):
                    if "/" in subclass:
                        if set(label) <= set(subclass):
                            k = i
                            break
                if k == -1:
                    if "결심3" in label:
                        label = "결심4"
                    if "번째" in label:
                        if "여덟" in label:
                            label = "8번째"
                        if "열" in label:
                            label = "10번째"
                        if "열두" in label:
                            label = "12번째"
                        if "아홉" in label:
                            label = "9번째"
                    k = self.classes.index(label)
            temp_list.append(k)
        labels = torch.tensor(temp_list).to("cuda")
        z = self.model(x, mask)
        z, labels = self.remove_nan(l, z, labels)
        z_tot = z
        
        acc = self.accuracy(z_tot, labels)
        self.test_acc_list.append(acc)
        self.test_acc_list.append(b)

    def on_test_epoch_end(self):
        test_acc_tensor = torch.tensor(self.test_acc_list[0::2])
        test_samples_tensor = torch.tensor(self.test_acc_list[1::2])
        self.test_acc_list = []
        average_test_acc = torch.sum(test_acc_tensor * test_samples_tensor) / torch.sum(test_samples_tensor)
        
        self.log('avg_test_acc', average_test_acc)

    def configure_optimizers(self):
        """
        optimizer = torch.optim.Adam(model.parameters(), 
                           lr=0.001,  # Initial learning rate
                           betas=(0.9, 0.999),  # Setting beta1 as momentum
                           weight_decay=0.0001)  # Setting the weight decay
    
        # Creating a learning rate scheduler that reduces the learning rate by a factor of 0.1 every 20 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)    
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        """
        if self.dataset == 'wlasl':
            lr = 3e-4
        if self.dataset == 'niasl':
            lr = 3e-5
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_train_loss"}   
    

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 16, dataset = "niasl", body12=True, aug_dataset_instruction=[], sub_sample=False, extended_length = 100, use_normalized=True, normalize_hand = True, x2 = False):
        super().__init__()
        self.dataset = dataset
        self.sub_sample = sub_sample
        self.extended_length = extended_length
        self.body12 = body12
        self.body12_index = torch.tensor([0,1,2,3,4,5,6,7,15,16,17,18])
        self.use_normalized = use_normalized
        self.normalize_hand = normalize_hand
        self.normalize = not use_normalized
        self.x2 = x2
        self.save_hyperparameters()
        if self.dataset == "niasl":
            if os.environ['CUDA_VISIBLE_DEVICES'] == '2':
                self.train_morp = '/data/sl_datasets/niasl20/nia20_lmdb_train_normalize_neo_54_0_0.5'
                self.val_morp = '/data/sl_datasets/niasl20/nia20_lmdb_test_normalize_neo_54_0_0.5'
                self.test_morp = '/data/sl_datasets/niasl20/nia20_lmdb_test_normalize_neo_54_0_0.5'
            elif os.environ['CUDA_VISIBLE_DEVICES'] == '3':
                self.train_morp = '/data/sl_datasets/niasl20/nia20_lmdb_train_normalize_neo_54_0_0.5_surplus'
                self.val_morp = '/data/sl_datasets/niasl20/nia20_lmdb_test_normalize_neo_54_0_0.5_surplus'
                self.test_morp = '/data/sl_datasets/niasl20/nia20_lmdb_test_normalize_neo_54_0_0.5_surplus'
            self.batch_size = batch_size
            self.aug_train_morp_list = []
            for (method, intensity) in aug_dataset_instruction:
                temp = "/data/sl_datasets/niasl20/nia20_lmdb_train_"+method
                aug_data_directory = temp if intensity == "None" else temp+"_"+intensity
                print(aug_data_directory)
                self.aug_train_morp_list.append(aug_data_directory)
        elif self.dataset == "wlasl":
            print(self.use_normalized, self.normalize_hand, self.x2)
            if self.x2:
                self.train_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_2x_normalize_neo_54_hand_0_0.5"
                self.val_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_val_normalize_neo_54_hand_0_0.5"
                self.test_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_neo_54_hand_0_0.5"
            elif self.use_normalized == True and self.normalize_hand == True and self.body12 == False:
                print("use already hand normalized data")
                self.train_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_neo_hand_0_0.5"
                self.val_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_val_normalize_neo_hand_0_0.5"
                self.test_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_neo_hand_0_0.5"
            elif self.use_normalized == True and self.normalize_hand == True and self.body12 == True:
                print("use already hand normalized data")
                self.train_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_neo_54_hand_0_0.5"
                self.val_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_val_normalize_neo_54_hand_0_0.5"
                self.test_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_neo_54_hand_0_0.5"
            elif self.use_normalized == True and ((not self.normalize_hand) == True):
                print("use already normalized data")
                self.train_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_normalize_0.5"
                self.val_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_val_normalize_0.5"
                self.test_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_normalize_0.5"
            elif self.use_normalized == False:
                print("use non-normalized data")
                self.train_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_original"
                self.val_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_val_original"
                self.test_morp = "/data/sl_datasets/wlasl/wlasl100_lmdb_test_original"
            self.batch_size = batch_size
            self.aug_train_morp_list = []
            
            for (method, intensity) in aug_dataset_instruction:
                temp = "/data/sl_datasets/wlasl/wlasl100_lmdb_train_"+method
                aug_data_directory = temp if intensity == "None" else temp+"_"+intensity
                print(aug_data_directory)
                self.aug_train_morp_list.append(aug_data_directory)
            
        self.save_hyperparameters({'datasets': [self.train_morp, self.val_morp, self.test_morp] + self.aug_train_morp_list})
                
            

    def setup(self, stage=None):
        if self.dataset == "niasl":
            org_train = NiaSLDataset20(self.train_morp, sub_sample=self.sub_sample, padding = False, crop = False, normalize = self.normalize, normalize_hand = self.normalize_hand)
            aug_train_list = []
            for aug_train_morp in self.aug_train_morp_list:
                aug_train_list.append(NiaSLDataset20(aug_train_morp, sub_sample=self.sub_sample, padding = False, crop = False, normalize = self.normalize, normalize_hand = self.normalize_hand))
            self.nia_val = NiaSLDataset20(self.val_morp, sub_sample=self.sub_sample, padding=False, crop = False, normalize = self.normalize, normalize_hand = self.normalize_hand)
            self.nia_test = self.nia_val
            self.nia_train = torch.utils.data.ConcatDataset([org_train] + aug_train_list)
        elif self.dataset == "wlasl":
            org_train = WlaSLDataset(self.train_morp, sub_sample=self.sub_sample, normalize = self.normalize, padding = False, extended_length = self.extended_length, normalize_hand = self.normalize_hand)
            aug_train_list = []
            for aug_train_morp in self.aug_train_morp_list:
                aug_train_list.append(WlaSLDataset(aug_train_morp, sub_sample=self.sub_sample, normalize = self.normalize, padding = False, extended_length = self.extended_length, normalize_hand = self.normalize_hand))
                print(aug_train_list[-1])
            self.nia_val = WlaSLDataset(self.val_morp, sub_sample=self.sub_sample, normalize = self.normalize, padding=False, extended_length = self.extended_length, normalize_hand = self.normalize_hand)
            self.nia_test = WlaSLDataset(self.test_morp, sub_sample=self.sub_sample, normalize = self.normalize, padding=False, extended_length = self.extended_length, normalize_hand = self.normalize_hand)
            self.nia_train = torch.utils.data.ConcatDataset([org_train] + aug_train_list)
    
    def custom_collate(self, batch):
        input_list = [item['keypoint'] for item in batch]
        labels = [item['gloss'] for item in batch]
        len_list = [item['keypoint'].shape[0] for item in batch]
        input_list= pad_sequence(input_list, batch_first=True, padding_value=0.)
        if self.body12 and (not input_list.shape[2]== 54):
            index = torch.cat((self.body12_index, torch.arange(25, 67)))
            input_list = input_list.index_select(2, index)
        if isinstance(batch[0]["index"], str):
            return {'keypoint': input_list.contiguous(),
                'gloss': labels,
                'len_list': len_list}
        else:
            index_list = [item["index"] for item in batch]
            return {'keypoint': input_list.contiguous(),
                'gloss': labels,
                'len_list': len_list,
                'index_list': index_list}
    def train_dataloader(self):
        return DataLoader(self.nia_train, batch_size = self.batch_size, num_workers=16, collate_fn=self.custom_collate, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.nia_val, batch_size = self.batch_size, num_workers=16, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return DataLoader(self.nia_test, batch_size = self.batch_size, num_workers=16, collate_fn=self.custom_collate)
    
def get_arguments():
    return 



if __name__=="__main__":
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("-aug_method", dest="aug_method", nargs='+', action="store", default=[])
    parser.add_argument("-model", dest="model", action="store", default="encoder")
    parser.add_argument("-resume", dest="resume", action="store", default="False")
    parser.add_argument("-test_only", dest="test_only", action="store", type=int, default=-1)
    parser.add_argument("-extended_length", dest="extended_length", action="store", type=int, default=100)
    parser.add_argument("-sub_sample", dest="sub_sample", action="store", default="False")
    parser.add_argument("-dataset", dest="dataset", action="store", default="niasl")
    parser.add_argument("-body12", dest="body12", action="store", default="True")
    parser.add_argument("-use_normalized", dest="use_normalized", action="store", default="True")
    parser.add_argument("-normalize_hand", dest="normalize_hand", action="store", default="True")
    args = parser.parse_known_args()[0]
    toBool = {'True': True,'False': False}
    args.resume = toBool[args.resume]
    args.sub_sample = toBool[args.sub_sample]
    args.body12 = toBool[args.body12]
    args.use_normalized = toBool[args.use_normalized]
    args.normalize_hand = toBool[args.normalize_hand]
    print(args.body12, args.normalize_hand, args.sub_sample, args.resume)
    
    if args.dataset == "niasl":
        max_epochs = 150
        with open('/home/ajkim/kslt/slr/NIA_classes3.json','r') as f:
            dict_json = json.load(f)
            word_dict = dict_json[0].keys()
            classes = list(word_dict)
            num_classes = len(classes)
            print("num_classes: %d"%num_classes)
    elif args.dataset == "wlasl":
        max_epochs = 200
        with open('/home/ajkim/kslt/slr/WLASL100_GLOSSES.json','r') as f:
            dict_json = json.load(f)
            word_vocab = vocab(dict_json, specials=["<unk>"])
            word_dict = word_vocab.get_stoi()
            classes = list(word_dict)
            num_classes = len(classes)
            print("num_classes: %d"%num_classes)
    #profiler1 = AdvancedProfiler(filename="profiler_output")
    #profiler2 = PyTorchProfiler(filename="profiler_pytorch")

    custom_callback = []

    logger = TensorBoardLogger("slr_logs", name= args.dataset)

    checkpoint_callback = ModelCheckpoint(filename="%s-(%s)-{v_num}-{epoch}-{val_acc_step:.3f}"%(args.model, ' '.join(args.aug_method)), save_top_k=1, monitor="val_acc_step", mode="max")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    custom_callback.append(checkpoint_callback)
    custom_callback.append(lr_monitor)
    
    if args.resume == True:
        trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, devices=1, resume_from_checkpoint='/home/ajkim/kslt/lightning_logs/version_204/checkpoints/epoch=146-step=1314180.ckpt') #, profiler=profiler2)
    elif not (args.test_only == -1):
        trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, devices=1, logger = False, callbacks=custom_callback) #, profiler=profiler2)
    else:
        trainer = pl.Trainer(accelerator="gpu", max_epochs=max_epochs, devices=1, logger = logger, callbacks=custom_callback) #, profiler=profiler2)
    
    aug_method = list(zip(args.aug_method[0::2], args.aug_method[1::2]))
    print(aug_method)
        
    model = TrainerModule(aug_method = aug_method, classes = classes, dataset = args.dataset, model = args.model, body12=args.body12)
    
    data = DataModule(dataset=args.dataset, body12=args.body12, aug_dataset_instruction=aug_method, sub_sample=args.sub_sample, extended_length=args.extended_length, use_normalized=args.use_normalized, normalize_hand=args.normalize_hand, x2 = (args.model == "weight_augment"))
    
    if (args.test_only == -1):
        trainer.fit(model, data)
        directory = "/home/ajkim/kslt/slr_logs/%s/version_%d/checkpoints/" %(args.dataset, trainer.logger.version)
    else:
        directory = "/home/ajkim/kslt/slr_logs/%s/version_%d/checkpoints/" %(args.dataset, args.test_only)
    print(directory)
    ckpt = os.path.join(directory, os.listdir(directory)[0])

    model = TrainerModule.load_from_checkpoint(ckpt, aug_method = aug_method, classes = classes, dataset = args.dataset, model = args.model, body12=args.body12)
    trainer.test(model, data)
    if not args.test_only == -1:
        import lmdb_maker
        import debug
        def load_model(logger_version):
            directory = "/home/ajkim/kslt/slr_logs/%s/version_%d/checkpoints/" %(args.dataset, logger_version)
            ckpt = os.path.join(directory, os.listdir(directory)[0])
            return TrainerModule.load_from_checkpoint(ckpt, aug_method = aug_method, classes = classes, dataset = args.dataset, model = args.model, body12=args.body12)
        def testing_function(i, data_split = 'test', k=5, verbose = True, func = (lambda x:x), testing_model = None, testing_data = data):
            if testing_model == None:
                testing_model = model
            if isinstance(testing_data, DataModule):
                if data_split == 'test':
                    datai = testing_data.nia_test[i]
                if data_split == 'train':
                    datai = testing_data.nia_train[i]
                output = testing_model.model(data.custom_collate([datai])['keypoint'], gen_src_mask(datai['keypoint'].shape[0],[datai['keypoint'].shape[0]]))
            else:
                datai = testing_data[i]
                output = testing_model.model(data.custom_collate([datai])['keypoint'], gen_src_mask(datai['keypoint'].shape[0],[datai['keypoint'].shape[0]]))
            datai['keypoint'] = func(datai['keypoint'])
            topk = torch.topk(output, k)
            output_results = []
            for j in range(k):
                output_results.append(classes[int(topk[1][0, j])])
            if verbose:
                print(topk)
                print(datai['gloss'], output_results)
            return datai['gloss'], output_results
        def test_noisy_inputs(i, noise=0.01, k=10, specific_model=model):
            if isinstance(noise, float):
                testing_function(i, data_split='test', k=k, verbose = True, func = lambda x:lmdb_maker.gaussian_helper_function(x.unsqueeze(0), noise, [True, True, True]).squeeze(0), testing_model = specific_model)
            if isinstance(noise, list):
                testing_function(i, data_split='test', k=k, verbose = True, func = lambda x:lmdb_maker.gaussian_helper_function(x.unsqueeze(0), {'POSE':noise[0], 'RHAND':noise[1], 'LHAND':noise[2]}, [True, True, True]).squeeze(0), testing_model = specific_model)
        count = 0
        occ30_count = 0
        occ50_count = 0
        occ70_count = 0
        occ99_count = 0
        total = len(data.nia_test)
        heavily_occluded_30 = [2,5,6,7,11,13,24,25,26,28,29,30,32,33,35,36,37,38,39,46,47,48,49,55,56,59,60,62,65,66,67,69,70,71,72,73,74,75,81,84,85,86,88,95,98,99,101,103,105,110,114,115,118,121,122,123,129,131,132,138,139,140,141,143,145,148,149,151,153,154,156,158,160,171,172,174,175,176,186,187,188,192,203,204,208,210,211,212,215,222,224,225,233,234,235,236,237,238,240,245,249,250,253,255,256,257]
        heavily_occluded_50 = [5,11,13,24,25,26,28,29,30,32,33,35,36,37,38,39,46,47,48,49,56,59,62,66,67,69,70,71,72,73,74,75,81,85,86,88,95,99,101,103,105,110,114,115,121,122,123,129,131,138,139,141,143,145,148,149,151,153,154,156,160,172,174,175,176,186,187,188,192,203,204,211,222,224,225,233,234,237,238,245,249,250,255,256]
        heavily_occluded_70 = [5,11,13,26,28,29,30,32,33,35,36,37,39,46,47,49,56,59,62,66,67,70,71,72,73,74,75,81,86,95,99,101,103,105,110,114,115,121,122,123,129,131,138,139,141,143,148,149,151,154,160,172,174,175,176,186,187,188,203,204,211,237,238,245,249,250,255,256]
        heavily_occluded_99 = [5,13,26,28,30,32,35,36,37,39,46,47,49,56,59,62,66,67,70,73,74,75,81,99,101,103,110,115,121,129,131,138,143,148,151,160,174,175,176,186,187,188,211,237,238,245,249,250,255]
        for i in range(total):
            tgt, res = testing_function(i, 'test', 5, False, testing_data = data.nia_test)
            if tgt == res[0]:
                count += 1
            if tgt == res[0] and i in heavily_occluded_30:
                occ30_count += 1
            if tgt == res[0] and i in heavily_occluded_50:
                occ50_count += 1
            if tgt == res[0] and i in heavily_occluded_70:
                occ70_count += 1
            if tgt == res[0] and i in heavily_occluded_99:
                occ99_count += 1
        print(count/total)
        print(occ30_count/len(heavily_occluded_30))
        print(occ50_count/len(heavily_occluded_50))
        print(occ70_count/len(heavily_occluded_70))
        print(occ99_count/len(heavily_occluded_99))
        import IPython; IPython.embed(); exit(1)
        
import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
import soundfile as sf
import os
import glob

class PreDataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        # self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type
        self.audios_paths = protocol_file_path

    def __len__(self):
        return len(self.audios_paths)

    def __getitem__(self, index):
        data_file_path = self.audios_paths[index]
        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = data_file_path.split('/')[-2].split('_')[0]
            label = label_encode(label)
            sub_class= label_encode(label) # trick
            return sample, label, sub_class
        
        if self.data_type == 'CQT':
            sample, _ = sf.read(data_file_path)
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = data_file_path.split('/')[-2].split('_')[0]
            label = label_encode(label)
            sub_class= label_encode(label) # trick
            return sample, label, sub_class
        
    def collate_fn(self, batch):
        # 获取批次中所有音频样本的长度
        lengths = [sample.size(-1) for sample, _, _ in batch]
        
        # 找到最长的音频样本长度
        max_length = max(lengths)
        
        # 填充所有样本,使其长度等于最长的长度
        padded_samples = []
        labels = []
        sub_classes = []
        for sample, label, sub_class in batch:
            padded_sample = torch.zeros(1, max_length, dtype=sample.dtype)
            padded_sample[0, :sample.size(-1)] = sample.squeeze(0)
            padded_samples.append(padded_sample)
            labels.append(label)
            sub_classes.append(sub_class)
        
        # 将填充后的样本堆叠为一个批次张量
        padded_samples = torch.cat(padded_samples, dim=0)
        labels = torch.stack(labels)
        sub_classes = torch.stack(sub_classes)
        
        return padded_samples, labels, sub_classes
    
    def get_weights(self):
        label_info = self.data_path.split('/')[-1]
        num_zero_class = int(label_info == 'bonafide')
        num_one_class = int(label_info == 'spoof')
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


class PrepASV19Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type
        self.train_paths = glob.glob(os.path.join(self.data_path, "*.wav"))

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.flac')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

        if self.data_type == 'CQT':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 4]
        num_zero_class = (label_info == 'bonafide').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights
    
    def collate_fn(batch):
        return batch
class PrepASV15Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.wav')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 3]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            return sample, label, sub_class

        if self.data_type == 'CQT':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 3]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 3]
        num_zero_class = (label_info == 'human').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights

    def collate_fn(batch):
        return batch


def label_encode(label):
    if label == 'bonafide':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'human':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == "real":
        label = torch.tensor(0, dtype=torch.int64)
    else:
        label = torch.tensor(1, dtype=torch.int64)
    return label


def sub_class_encode_19(label):
    if label == '-':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'A01':
        label = torch.tensor(1, dtype=torch.int64)
    elif label == 'A02':
        label = torch.tensor(2, dtype=torch.int64)
    elif label == 'A03':
        label = torch.tensor(3, dtype=torch.int64)
    elif label == 'A04':
        label = torch.tensor(4, dtype=torch.int64)
    elif label == 'A05':
        label = torch.tensor(5, dtype=torch.int64)
    elif label == 'A06':
        label = torch.tensor(6, dtype=torch.int64)
    elif label == 'A07':
        label = torch.tensor(7, dtype=torch.int64)
    elif label == 'A08':
        label = torch.tensor(8, dtype=torch.int64)
    elif label == 'A09':
        label = torch.tensor(9, dtype=torch.int64)
    elif label == 'A10':
        label = torch.tensor(10, dtype=torch.int64)
    elif label == 'A11':
        label = torch.tensor(11, dtype=torch.int64)
    elif label == 'A12':
        label = torch.tensor(12, dtype=torch.int64)
    elif label == 'A13':
        label = torch.tensor(13, dtype=torch.int64)
    elif label == 'A14':
        label = torch.tensor(14, dtype=torch.int64)
    elif label == 'A15':
        label = torch.tensor(15, dtype=torch.int64)
    elif label == 'A16':
        label = torch.tensor(16, dtype=torch.int64)
    elif label == 'A17':
        label = torch.tensor(17, dtype=torch.int64)
    elif label == 'A18':
        label = torch.tensor(18, dtype=torch.int64)
    elif label == 'A19':
        label = torch.tensor(19, dtype=torch.int64)
    return label


def sub_class_encode_15(label):
    if label == 'human':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'S1':
        label = torch.tensor(1, dtype=torch.int64)
    elif label == 'S2':
        label = torch.tensor(2, dtype=torch.int64)
    elif label == 'S3':
        label = torch.tensor(3, dtype=torch.int64)
    elif label == 'S4':
        label = torch.tensor(4, dtype=torch.int64)
    elif label == 'S5':
        label = torch.tensor(5, dtype=torch.int64)
    elif label == 'S6':
        label = torch.tensor(6, dtype=torch.int64)
    elif label == 'S7':
        label = torch.tensor(7, dtype=torch.int64)
    elif label == 'S8':
        label = torch.tensor(8, dtype=torch.int64)
    elif label == 'S9':
        label = torch.tensor(9, dtype=torch.int64)
    elif label == 'S10':
        label = torch.tensor(10, dtype=torch.int64)
    return label

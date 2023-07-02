# dataset
from torch.utils.data import Dataset, DataLoader
import torchaudio
import random
from tools.processing import *
import numpy as np

class MLTDataset(Dataset):
    def __init__(self, opt=None, train_mode=True):
        """
        Args:
            utt2wav (list): 
            utt2spk (dict): 
            opt :
        """
        self.opt = opt
        self.train_mode = train_mode
        self.max_frames = opt.max_frames
        self.fs = opt.fs
        if train_mode:
            self.noise_list = opt.noise_list
            self.utt2wav = opt.utt2wav
            self.spk2int = opt.spk2int
            self.data_wavaug = opt.data_wavaug
            # self.utt2spk = {line.split()[0]:opt.spk2int[line.split()[1]] for line in open('%s/utt2spk' % opt.train_dir)}
            self.utt2spk = {line.split()[0]:opt.spk2int[line.split()[1]] for line in open('%s/train_1m_utt2spk' % opt.train_dir)}
        else:
            self.utt2wav = opt.utt2wav_val

    def __len__(self):
        return len(self.utt2wav)
    
    def __getitem__(self, idx):
        utt, filename = self.utt2wav[idx]
        # signal, sample_rate = torchaudio.load(filename.replace('/root/智能家居场景说话人识别挑战赛','../../'))
        filename = filename.replace('/root/teamshare','..')
        signal = load_wav(filename, max_frames=self.max_frames, fs=self.fs, train_mode=self.train_mode)
        if self.train_mode:           
            if self.data_wavaug:
                aug_type = random.sample(['noise', 'rir', 'clean', 'clean'],1)[0]
                if aug_type == 'noise':
                    signal = addnoise(signal,self.noise_list,max_frames=self.max_frames)
                elif aug_type == 'rir':
                    signal = addreverberate(signal,self.noise_list,max_frames=self.max_frames)
                    
            signal = mean_std_norm_1d(signal)
            feat = signal
                    
            return feat,self.utt2spk[utt]
        
        else:
            signal = mean_std_norm_1d(signal)
            feat = signal
            
            return feat,utt
        
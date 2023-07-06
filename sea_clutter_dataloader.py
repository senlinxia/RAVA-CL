from torch.utils.data import Dataset
from config import cfg
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from scipy import signal


def add_noise(Echo):
    noise_real = cfg.GAUSSIAN.TRANSFORM_NOISE * np.random.randn(
        cfg.SEA_CLUTTER.SMALL_LEN)
    noise_imag = cfg.GAUSSIAN.TRANSFORM_NOISE * np.random.randn(
        cfg.SEA_CLUTTER.SMALL_LEN)
    noise = noise_real + 1j * noise_imag
    E=noise+Echo

    return E

def crop(Echo):
    if cfg.SEA_CLUTTER.SMALL_LEN==64:
        len_=50
    elif cfg.SEA_CLUTTER.SMALL_LEN==128:
        len_=100
    elif cfg.SEA_CLUTTER.SMALL_LEN==32:
        len_ = 25
    elif cfg.SEA_CLUTTER.SMALL_LEN==16:
        len_ = 12
    start=random.randint(0,cfg.SEA_CLUTTER.SMALL_LEN-len_)
    echo=Echo[start:start+len_]
    e=signal.resample(echo, cfg.SEA_CLUTTER.SMALL_LEN)
    return e

def flip(Echo):
    e=Echo[::-1]
    ee=e.copy()

    return ee

def org(Echo):
    return Echo

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, transforms, n_views=2):
        self.transforms = transforms
        self.n_views = n_views

    def __call__(self, x):
        f=[]

        for i in range(self.n_views):
            t = random.choice(self.transforms)
            f.append(abs(t(x)[None,:]))
        return f


def label2target(label,win_size):
    poss=np.where(label!=0)[0]

    target = np.zeros((2, label.shape[0]))

    window=np.ones(win_size)

    temp=np.zeros(label.shape)

    for p in poss:
        temp[p-win_size//2:p+win_size//2+1]=window

    target[0]=1-temp
    target[1]=temp

    return target

class SeaClutterDataset():
    def __init__(self,whole=False,distribution=None):
        super(SeaClutterDataset, self).__init__()
        self.target_datapack = []
        self.none_datapack=[]
        self.datapack=[]
        self.whole=whole
        transforms=[org,add_noise,flip,crop]

        self.transform=ContrastiveLearningViewGenerator(transforms)
        if cfg.SEA_CLUTTER.TRAIN:
            if cfg.SEA_CLUTTER.SMALL:

                if not distribution:
                    target_pack = np.load(os.path.join(cfg.DATA.GEN_S_TRAIN_DATA,
                                         'small_data_with_target_%d_%d_%d.npy' % (
                                         cfg.SEA_CLUTTER.SNR_MIN, cfg.SEA_CLUTTER.SNR_MAX,
                                         cfg.SEA_CLUTTER.SMALL_LEN)),allow_pickle=True)
                    noise_pack = np.load(os.path.join(cfg.DATA.GEN_S_TRAIN_DATA,
                                         'small_data_without_target_%d_%d_%d.npy' % (
                                             cfg.SEA_CLUTTER.SNR_MIN,
                                             cfg.SEA_CLUTTER.SNR_MAX,
                                             cfg.SEA_CLUTTER.SMALL_LEN)),
                                          allow_pickle=True)
                else:
                    target_pack = np.load(
                        os.path.join(cfg.DATA.GEN_SIM_S_TRAIN_DATA,
                                     distribution,
                                     'small_data_with_target_%d_%d_%d.npy' % (
                                         cfg.SEA_CLUTTER.SNR_MIN,
                                         cfg.SEA_CLUTTER.SNR_MAX,
                                         cfg.SEA_CLUTTER.SMALL_LEN)),
                        allow_pickle=True)
                    noise_pack = np.load(os.path.join(cfg.DATA.GEN_SIM_S_TRAIN_DATA,distribution,
                                 'small_data_without_target_%d_%d_%d.npy' % (
                                     cfg.SEA_CLUTTER.SNR_MIN,
                                     cfg.SEA_CLUTTER.SNR_MAX,
                                     cfg.SEA_CLUTTER.SMALL_LEN)),
                                         allow_pickle=True)

            else:
                target_pack = np.load(os.path.join(cfg.DATA.GEN_G_TRAIN_DATA,
                                                   'data_with_target_%d_%d_%d.npy' % (
                                                   cfg.SEA_CLUTTER.SNR_MIN,
                                                   cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.LEN)),
                                      allow_pickle=True)
                noise_pack = np.load(os.path.join(cfg.DATA.GEN_G_TRAIN_DATA,
                                                  'data_without_target_%d_%d_%d.npy' % (
                                                      cfg.SEA_CLUTTER.SNR_MIN,
                                                      cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.LEN)),
                                     allow_pickle=True)
        else:
            if cfg.SEA_CLUTTER.SMALL:
                target_pack = np.load(os.path.join(cfg.DATA.GEN_S_TEST_DATA , 'test_small_data_with_target_%d_%d_%d.npy'%(cfg.SEA_CLUTTER.SNR_MIN,cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.SMALL_LEN)),allow_pickle=True)
                noise_pack = np.load(os.path.join(cfg.DATA.GEN_S_TEST_DATA,
                                                   'test_small_data_without_target_%d_%d_%d.npy' % (
                                                   cfg.SEA_CLUTTER.SNR_MIN,
                                                   cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.SMALL_LEN)),
                                      allow_pickle=True)
            else:
                target_pack = np.load(os.path.join(cfg.DATA.GEN_G_TEST_DATA,
                                                   'data_with_target_%d_%d_%d.npy' % (
                                                   cfg.SEA_CLUTTER.SNR_MIN,
                                                   cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.LEN)),
                                      allow_pickle=True)
                noise_pack = np.load(os.path.join(cfg.DATA.GEN_G_TEST_DATA,
                                                  'data_without_target_%d_%d_%d.npy' % (
                                                      cfg.SEA_CLUTTER.SNR_MIN,
                                                      cfg.SEA_CLUTTER.SNR_MAX,cfg.SEA_CLUTTER.LEN)),
                                     allow_pickle=True)


        for i in range(len(target_pack)):
            this_pack={}
            Echo = target_pack[i]['Echo']
            Echo=Echo/np.mean(abs(Echo))
            label=target_pack[i]['label']
            echos=self.transform(Echo)

            snr=target_pack[i]['snr']
            this_pack['echo_r'] = Echo.real[None,:]
            this_pack['echo_i'] = Echo.imag[None,:]
            this_pack['pt'] = target_pack[i]['pt']
            this_pack['label']=target_pack[i]['label']
            target = label2target(label, cfg.SEA_CLUTTER.WIN_SIZE)
            this_pack['target'] =target
            this_pack['cls_label'] = np.asarray([0, 1])
            this_pack['cls_target'] = np.asarray([1])
            this_pack['echos']=echos

            self.datapack.append(this_pack)

        cnt=0

        if self.whole==False:
            while cnt < len(target_pack):
                i=random.randint(0,len(noise_pack)-1)
                this_pack={}
                Echo = noise_pack[i]['Echo']
                Echo = Echo / np.mean(abs(Echo))
                label=noise_pack[i]['label']
                echos = self.transform(Echo)
                snr=noise_pack[i]['snr']
                this_pack['echo_r'] = Echo.real[None,:]
                this_pack['echo_i'] = Echo.imag[None,:]
                this_pack['echos'] = echos
                this_pack['pt'] = noise_pack[i]['pt']
                this_pack['label']=noise_pack[i]['label']
                # plt_ori(Echo.real, Echo.imag, label, this_pack['pt'])
                target = label2target(label, cfg.SEA_CLUTTER.WIN_SIZE)
                this_pack['target'] =target
                this_pack['cls_label'] = np.asarray([1, 0])
                this_pack['cls_target'] = np.asarray([0])

                self.datapack.append(this_pack)

                cnt+=1
                # self.datapack.append(this_pack)
        else:
            for i in range(len(noise_pack)):
                this_pack = {}
                Echo = noise_pack[i]['Echo']
                Echo = Echo / np.mean(abs(Echo))
                label = noise_pack[i]['label']
                echos = self.transform(Echo)
                snr = noise_pack[i]['snr']
                this_pack['echo_r'] = Echo.real[None, :]
                this_pack['echo_i'] = Echo.imag[None, :]
                this_pack['echos'] = echos
                this_pack['pt'] = noise_pack[i]['pt']
                this_pack['label'] = noise_pack[i]['label']
                # plt_ori(Echo.real, Echo.imag, label, this_pack['pt'])
                target = label2target(label, cfg.SEA_CLUTTER.WIN_SIZE)
                this_pack['target'] = target
                this_pack['cls_label'] = np.asarray([1, 0])
                this_pack['cls_target'] = np.asarray([0])

                self.datapack.append(this_pack)
        # print(len(self.datapack))
    def __len__(self):

        return len(self.datapack)

    def __getitem__(self, index):
        return self.datapack[index]



if __name__ == '__main__':
    data = SeaClutterDataset(whole=True)

from copyreg import pickle
import torch
from model import Generator
from config import CFG
import numpy as np
from math import ceil

from torch import nn

import pickle

CFG.init(dim_neck = 32, freq = 32)

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = "cpu"
G = Generator().eval().to(device)
mod = torch.load('generator.ckpt')
G.load_state_dict(mod)


#a = pickle.load(open("/Users/eleonoredebokay/Downloads/metadata.pkl", 'rb'))
#x, y, z = a[0]


spec = np.load("p225_003.npy")
emb1 = np.load("spkr1.npy")
emb2 = np.load("spkr2.npy")

#print("IN SHAPE", spec.shape, emb1.shape, emb2.shape)

# Prepro spec
spec, len_pad = pad_seq(spec)
spec = torch.from_numpy(spec[np.newaxis, :, :])#.to(device)

## Prepro embedding
emb1 = torch.from_numpy(emb1[np.newaxis, :])#.to(device)
emb2 = torch.from_numpy(emb2[np.newaxis, :])#.to(device)

spect_vc = []
with torch.no_grad():
    _, psnt, _ = G(spec, emb1, emb2)
    if len_pad == 0:
        trgt_uttr = psnt[0, 0, :, :].cpu().numpy()
    else:
        trgt_uttr = psnt[0, 0, :-len_pad, :].cpu().numpy()
    #spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), trgt_uttr) )
    spect_vc.append( ('contentXspeaker', trgt_uttr) )
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
    

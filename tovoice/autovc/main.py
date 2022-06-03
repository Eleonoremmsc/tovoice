from copyreg import pickle
import torch
import numpy as np
from math import ceil

from autovc import Generator # -> can I remove ?

device = "cpu"
G = Generator().eval().to(device)
mod = torch.load('generator.ckpt')
G.load_state_dict(mod)

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad


def auto_main(spec, emb1, emb2):
    # Preprocess spec
    spec, len_pad = pad_seq(spec)
    spec = torch.from_numpy(spec[np.newaxis, :, :])

    spect_vc = []
    with torch.no_grad():
        _, psnt, _ = G(spec, emb1, emb2)
        trgt_uttr = psnt[0, 0, :-len_pad, :].cpu().numpy()
        spect_vc.append( ('contentXspeaker', trgt_uttr) )


    with open('results.pkl', 'wb') as handle:
        pickle.dump(spect_vc, handle)





















    # if len_pad == 0:
        #     trgt_uttr = psnt[0, 0, :, :].cpu().numpy()
    # else:
        # trgt_uttr = psnt[0, 0, :-len_pad, :].cpu().numpy()
    # spect_vc.append( ('contentXspeaker', trgt_uttr) )

    ## Prepro embedding
    #emb1 = torch.from_numpy(emb1[np.newaxis, :])#.to(device)
    #emb2 = torch.from_numpy(emb2[np.newaxis, :])#.to(device)

    #spec = np.load("p225_003.npy")
    #emb1 = np.load("spkr1.npy")
    #emb2 = np.load("spkr2.npy")

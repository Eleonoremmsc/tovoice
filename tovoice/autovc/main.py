import os
import pickle
import torch
import numpy as np

from copyreg import pickle
from math import ceil
from pathlib import Path

from autovc import Generator, get_generator
from preprocessing import pad_seq


def auto_main(spec_file, emb1_name, emb2_name):

    SOURCE_FILE = Path(__file__).resolve()
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent
    spec = np.load(ROOT_DIR /"data/spectrograms"/emb1_name/ spec_file)
    emb1 = torch.load(ROOT_DIR /"data/speaker-embeddings"/ emb1_name)
    emb2 = torch.load(ROOT_DIR /"data/speaker-embeddings"/ emb2_name)

    G = get_generator()

    #G = Generator().eval().to("cpu")
    #mod = torch.load(ROOT_DIR.parent/'generator_syn.ckpt')
    #G.load_state_dict(mod)


    # Preprocess spec
    spec, len_pad = pad_seq(spec)
    spec = torch.from_numpy(spec[np.newaxis, :, :])

    spect_vc = []
    with torch.no_grad():
        _, psnt, _ = G(spec, emb1, emb2)
        trgt_uttr = psnt[0, 0, :-len_pad, :].cpu().numpy()
        spect_vc.append(('{}x{}'.format(emb1_name, emb2_name), trgt_uttr))

    with open(ROOT_DIR/'data/autovc_results/{}_x_{}_x_{}.pkl'.format(emb1_name, spec_file, emb2_name), 'wb') as handle:
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

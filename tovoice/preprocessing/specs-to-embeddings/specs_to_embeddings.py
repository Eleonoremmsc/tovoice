
import os
from model_bl import D_VECTOR
import numpy as np
import torch

# fonction qui génère directement un modele d'embedding préentraîné :
def load_model() :

    model = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cpu()
    pre_trained_weights = torch.load('pre_trained_embedding_model.ckpt',
                                     map_location=torch.device('cpu'))

    model.load_state_dict(pre_trained_weights)
    return model


# définition des dossier source (specs) et cible (embeddings)
specs_dir = '../../data/spectograms'
emb_dir ='../../data/embeddings'


# fonction qui trouve les différents speakers dans le répertoire source
def get_speakers() :
    _, spkrs_list, _ = next(os.walk(specs_dir))
    return spkrs_list


#fonction qui génére un embedding vector pour un speaker donné
def  get_embedding(spkr_name) :

    #chargment du modéle
    model = load_model()

    #définition de la liste des fichiers de spectrogrammes d'un speaker
    _, _, file_list = next(os.walk(os.path.join(specs_dir,spkr_name)))

    #définition de la taille std des extraits de voix pour générer les embedding
    len_crop = 128

    #génération d'une liste d'embeddings
    for i in file_list :
        embs=[]
        tmp = np.load(os.path.join(specs_dir, spkr_name, i))
        left = np.random.randint(0, tmp.shape[0]-len_crop)
        melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cpu()
        emb = model(melsp)
        embs.append(emb.detach().squeeze().cpu().numpy())

    #retourne la moyenne des embeddings
    return np.mean(embs, axis=0)


#fonction qui sauvegarde un embedding vector dans le dossier cible
def save_embedding(spkr_name) :
    spkr_embedding=get_embedding(spkr_name)

    np.save(os.path.join(emb_dir,spkr_name),spkr_embedding, allow_pickle=False)



if __name__ == '__main__':
    spkrs_list = get_speakers()
    for speaker in spkrs_list :
        save_embedding(speaker)

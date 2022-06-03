import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cpu")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    sf.write(name+'.wav', waveform, samplerate=16000)
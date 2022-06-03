import torch
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen


class Vocoder():
    
    def __init__(self, results_pathfile):
        self.spect_vc = pickle.load(open(results_pathfile, 'rb'))
        self.device = torch.device("cpu")
        self.model = build_model().to(self.device)
        self.checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location=self.device)
        self.model.load_state_dict(self.checkpoint["state_dict"])

    def generate_wav_file(self):
        for spect in self.spect_vc:
            name = spect[0]
            c = spect[1]
            waveform = wavegen(self.model, c=c)   
            sf.write(f"../data/outputs/{name}.wav", waveform, samplerate=16000)

if __name__ == "__main__":
    vocoder = Vocoder("../data/autovc_results/results.pkl")
    vocoder.generate_wav_file()

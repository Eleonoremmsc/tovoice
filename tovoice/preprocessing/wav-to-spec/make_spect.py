from pathlib import Path

from xml.etree.ElementInclude import default_loader

import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState


# 0. Retrieve speaker directory
def retrieve_dirpath(speaker):
    SOURCE_FILE = Path(__file__).resolve()
    SOURCE_DIR = SOURCE_FILE.parent.parent
    ROOT_DIR = SOURCE_DIR.parent
    speaker_wavs_dir = ROOT_DIR / "data" / "wavs" / speaker
    speaker_spects_dir = ROOT_DIR / "data" / "spectrograms" / speaker
    speaker_spects_dir.mkdir(exist_ok=True)
    return speaker_wavs_dir, speaker_spects_dir

# 1. Get the file paths to an included audio in a speaker dir

def get_wav_files(speaker_wav_dir):
    wav_files = []

    for f in speaker_wav_dir.rglob("*.wav"):
        wav_files.append(f)
    
    return wav_files

    
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
def generate_wave_forms(wav_files):
    wave_forms = []

    for wav_file in wav_files:
        # y, sr = librosa.load(wav_file)
        y, sr = sf.read(wav_file)
        wave_forms.append((y, sr))

    return wave_forms


#2.1 Apply the butterworth filter:
def wavs_to_butters(wave_forms, cutoff=30, fs=16_000, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    butter_files=[]
    for wave_form in wave_forms:
        y, sr = wave_form
        wav = signal.filtfilt(b, a, y) # remove drifting noise
        # wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06 # Add a little random noise for model roubstness
        butter_files.append((wav, sr))
    return butter_files


# 2.2 
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result) 

# 3. Generate a mel spec
def generate_mel_specs(butter_files):
    mel_specs = []

    for butter_file in butter_files:
        y, sr = butter_file
        # S = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        D = pySTFT(y).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)  
        mel_specs.append((S, sr))
    return mel_specs

    
# 4. Display mel-frequency spectrograms
def display_mel_specs(mel_specs):
    nb_cols = 5
    nb_rows = int(len(mel_specs) / nb_cols) + 1
    fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(30,50))

    for i, mel_spec in enumerate(mel_specs):
        row_index = i // nb_cols
        col_index = i % nb_cols
        ax = axes[row_index, col_index]
        S, sr = mel_spec
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')

    plt.show()
    
    
# 5. Save mel-frequency spectrograms in Data directory

def save_mel_specs(mel_specs, wav_files, speaker_spects_dir):
    for i, mel_spec in enumerate(mel_specs):
        filepath = speaker_spects_dir / wav_files[i].stem
        np.save(filepath, mel_spec[0].astype(np.float32), allow_pickle=False)


if __name__ == "__main__":
    speaker_wavs_dir, speaker_spects_dir = retrieve_dirpath("eva-longoria")
    specs_dir = '../../data/spectrograms'
    wav_files = get_wav_files(speaker_wavs_dir)
    wave_forms = generate_wave_forms(wav_files)
    butter_files = wavs_to_butters(wave_forms, cutoff=30, fs=16_000, order=5)
    mel_specs = generate_mel_specs(butter_files) 
    display_mel_specs(mel_specs)
    save_mel_specs(mel_specs, wav_files, speaker_spects_dir)

    
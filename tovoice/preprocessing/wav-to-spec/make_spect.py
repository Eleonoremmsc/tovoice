from pathlib import Path
from xml.etree.ElementInclude import default_loader

import librosa
import librosa.display

import matplotlib.pyplot as plt
import numpy as np

# 0. Retrieve file_path
def retrieve_filepath():
    SOURCE_FILE = Path(__file__).resolve()
    SOURCE_DIR = SOURCE_FILE.parent.parent
    ROOT_DIR = SOURCE_DIR.parent
    DATA_DIR = ROOT_DIR / "data"
    return DATA_DIR

# 1. Get the file path to an included audio in data
def get_wav_files(data_dir):
    wav_files = []

    for f in data_dir.rglob("*.wav"):
        wav_files.append(f)
    
    return wav_files

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
def generate_wave_forms(wav_files):
    wave_forms = []

    for wav_file in wav_files:
        y, sr = librosa.load(wav_files[0])
        wave_forms.append((y, sr))
    
    return wave_forms

# 3. Generate a mel spec
def generate_mel_specs(wave_forms):
    mel_specs = []

    for wave_form in wave_forms:
        y, sr = wave_form
        S = librosa.feature.melspectrogram(y=y, sr=sr)
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
def save_mel_specs(mel_specs, wav_files, data_dir):
    for i, mel_spec in enumerate(mel_specs):
        filepath = data_dir / "spectrograms" / wav_files[i].stem
        np.save(filepath, mel_spec[0])


if __name__ == "__main__":
    DATA_DIR = retrieve_filepath()
    wav_files = get_wav_files(DATA_DIR)
    wave_forms = generate_wave_forms(wav_files)
    mel_specs = generate_mel_specs(wave_forms) 
    # display_mel_specs(mel_specs)
    save_mel_specs(mel_specs, wav_files, DATA_DIR)
    
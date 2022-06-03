import subprocess

# prend un texte et génére un .wav via le shell (pip install tts nécessaire)
def text_to_speech(text,path='data/wavs/synthetic/speech.wav') :

    subprocess.run(["tts","--text",text,'--out_path',path])

import subprocess
from pathlib import Path
import pandas as pd

# prend un texte et génére un .wav via le shell (pip install tts nécessaire)
def text_to_speech(text,path='data/wavs/synthetic_demo/speech.wav') :
    SOURCE_FILE = Path(__file__).resolve()
    SOURCE_DIR = SOURCE_FILE.parent
    ROOT_DIR = SOURCE_DIR.parent
    subprocess.run(["tts","--text",text,'--out_path', ROOT_DIR / path])


#text_to_speech("Welcome to demoday", 'data/wavs/synthetic/demoday.wav')
#text_to_speech("we like text to voice", 'data/wavs/synthetic/we_like.wav')
#text_to_speech("here comes the sun", 'data/wavs/synthetic/sun.wav')
#text_to_speech("we will rock you", 'data/wavs/synthetic/rockyou.wav')
#import pandas as pd
#
#data = pd.read_csv("/Users/eleonoredebokay/Downloads/short_sentences.csv")
#sentence_a = data["sentenceA"].to_list()
#for a in sentence_a:
#    text_to_speech(a, 'data/wavs/synthetic/{}.wav'.format(sentence_a.index(a)))
#text_to_speech("Please call Stella", 'data/wavs/synthetic/stella.wav')

lsit = ["welcome to demoday",
        "where is bryan",
        "bryan is in the kitchen",
        "we love neural networks",
        "paris is a beautiful city",
        "data science is so cool"]

for a in lsit:
    text_to_speech(a, 'data/wavs/synthetic_demo/{}.wav'.format(lsit.index(a)))

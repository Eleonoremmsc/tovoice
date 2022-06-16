import sys
from pathlib import Path

from preprocessing import Preprocessor
from autovc import auto_main
from postprocessing import Vocoder

class Controller():
    def __init__(self, spec_file, speaker_name, style_name):
        self.spec_file = spec_file
        self.speaker_name = speaker_name
        self.style_name = style_name

        for _ in [speaker_name, style_name]:
           self.preprocessor = Preprocessor(_)
           self.preprocessor.preprocess()
        auto_main(spec_file, speaker_name, style_name)

        SOURCE_FILE = Path(__file__).resolve()
        SOURCE_DIR = SOURCE_FILE.parent

        self.vocoder = Vocoder(SOURCE_DIR/f"data/autovc_results/{self.speaker_name}_x_{self.spec_file}_x_{self.style_name}.pkl")
        self.vocoder.generate_wav_file()


if __name__ == "__main__":
    # speaker_name = "p226"
    # spec_file = "p226_024.npy"
    # style_name = "Jimmy_Fallon"
    
    speaker_name = sys.argv[1]
    spec_file = sys.argv[2]
    style_name = sys.argv[3]
    controller = Controller(spec_file, speaker_name, style_name)
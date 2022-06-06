import sys
from autovc import pad_seq, auto_main
from preprocessing import Preprocessor
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
        
        self.vocoder = Vocoder(f"data/autovc_results/{self.speaker_name}x{self.style_name}.pkl")
        #self.vocoder = Vocoder(f"data/autovc_results/results.pkl")
        self.vocoder.generate_wav_file()



if __name__ == "__main__":
    # speaker_name = "p225"
    # spec_file = "p225_003.npy"
    # style_name = "p225"
    
    speaker_name = sys.argv[1]
    spec_file = sys.argv[2]
    style_name = sys.argv[3]
    controller = Controller(spec_file, speaker_name, style_name)


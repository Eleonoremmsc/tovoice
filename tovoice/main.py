from autovc import pad_seq, auto_main
from preprocessing import Preprocessor




class Controller():
    def __init__(self, speaker_name, style_name):
        self.speaker_name = speaker_name
        self.style_name = style_name
        self.preprocessor = Preprocessor(speaker_name)
    
        #auto_main(spec_file, emb1_name, emb2_name)



if __name__ == "__main__":
    speaker_name = "p227"
    style_name = "p228"
    controller = Controller(speaker_name, style_name)
    # spec_file = 
    # emb1_name, emb2_name =  
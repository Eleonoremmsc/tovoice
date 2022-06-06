from wav_to_spec import MakeSpecs
from specs_to_embeddings.specs_to_embeddings import MakeEmbedding

class Preprocessor():
    def __init__(self, speaker_name):
        self.speaker_name = speaker_name
        
    def preprocess(self):
        make_spec = MakeSpecs(self.speaker_name)
        make_spec.generate_all_specs()


        make_emb = MakeEmbedding(self.speaker_name)
        make_emb.save_embedding()
        
if __name__ == "__main__":
    speaker_name = "p226"
    preprocessor = Preprocessor(speaker_name)
    preprocessor.preprocess()


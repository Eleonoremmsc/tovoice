from wav_to_spec import MakeSpecs
from specs_to_embeddings.specs_to_embeddings import save_embedding

speaker_name = "p225"

make_spec = MakeSpecs(speaker_name)
make_spec.generate_all_specs()

save_embedding(speaker_name)
from wav_to_spec import MakeSpecs
from specs_to_embeddings.specs_to_embeddings import MakeEmbedding

make_spec = MakeSpecs(speaker_name)
make_spec.generate_all_specs()


make_emb = MakeEmbedding(speaker_name)
make_emb.save_embedding(speaker_name)

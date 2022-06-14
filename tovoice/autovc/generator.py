import torch
from pathlib import Path
from autovc import Decoder, Encoder, Postnet


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, spc, emb_1, emb_2):

        # TODO Better name ? input_1 ?
        codes = self.encoder(spc, emb_1)

        if emb_2 is None:
            return torch.cat(codes, dim=-1)

        tmp = []
        for code in codes:
            tmp.append(
                code.unsqueeze(1).expand(-1, int(spc.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat(
            (code_exp, emb_2.unsqueeze(1).expand(-1, spc.size(1), -1)), dim=-1)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

def get_generator():

    SOURCE_FILE = Path(__file__).resolve()

    G = Generator().eval().to("cpu")
    # TODO Cleanup
    mod = torch.load(SOURCE_FILE.parent /'generator.ckpt')
    G.load_state_dict(mod)
    return G

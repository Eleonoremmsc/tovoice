from copyreg import pickle
import torch
from tovoice.model import Generator
from tovoice.config import CFG
import numpy as np
from math import ceil

from torch import nn

import pickle

device = "cpu"
G = Generator().eval().to(device)

mod = torch.load('autovc.ckpt', map_location=device)["model"]
opt = torch.load('autovc.ckpt', map_location=device)["optimizer"]

#import sys
#print(sys.getsizeof(mod))
#print(sys.getsizeof(opt))

with torch.no_grad():
    G.encoder.conv[0].weight.data = mod["encoder.convolutions.0.0.conv.weight"]
    G.encoder.conv[0].bias.data = mod["encoder.convolutions.0.0.conv.bias"]
    G.encoder.conv[1].weight.data = mod["encoder.convolutions.0.1.weight"]
    G.encoder.conv[1].bias.data = mod["encoder.convolutions.0.1.bias"]
    G.encoder.conv[1].running_mean.data = mod["encoder.convolutions.0.1.running_mean"]
    G.encoder.conv[1].running_var.data = mod["encoder.convolutions.0.1.running_var"]
    G.encoder.conv[1].num_batches_tracked.data = mod["encoder.convolutions.0.1.num_batches_tracked"]
    G.encoder.conv[3].weight.data = mod["encoder.convolutions.1.0.conv.weight"]
    G.encoder.conv[3].bias.data = mod["encoder.convolutions.1.0.conv.bias"]
    G.encoder.conv[4].weight.data = mod["encoder.convolutions.1.1.weight"]
    G.encoder.conv[4].bias.data = mod["encoder.convolutions.1.1.bias"]
    G.encoder.conv[4].running_mean.data = mod["encoder.convolutions.1.1.running_mean"]
    G.encoder.conv[4].running_var.data = mod["encoder.convolutions.1.1.running_var"]
    G.encoder.conv[4].num_batches_tracked.data = mod["encoder.convolutions.1.1.num_batches_tracked"]
    G.encoder.conv[6].weight.data = mod["encoder.convolutions.2.0.conv.weight"]
    G.encoder.conv[6].bias.data = mod["encoder.convolutions.2.0.conv.bias"]
    G.encoder.conv[7].weight.data = mod["encoder.convolutions.2.1.weight"]
    G.encoder.conv[7].bias.data = mod["encoder.convolutions.2.1.bias"]
    G.encoder.conv[7].running_mean.data = mod["encoder.convolutions.2.1.running_mean"]
    G.encoder.conv[7].running_var.data = mod["encoder.convolutions.2.1.running_var"]
    G.encoder.conv[7].num_batches_tracked.data = mod["encoder.convolutions.2.1.num_batches_tracked"]

    G.encoder.lstm.weight_ih_l0.data = mod["encoder.lstm.weight_ih_l0"]
    G.encoder.lstm.weight_hh_l0.data = mod["encoder.lstm.weight_hh_l0"]
    G.encoder.lstm.bias_ih_l0.data = mod["encoder.lstm.bias_ih_l0"]
    G.encoder.lstm.bias_hh_l0.data = mod["encoder.lstm.bias_hh_l0"]
    G.encoder.lstm.weight_ih_l0_reverse.data = mod["encoder.lstm.weight_ih_l0_reverse"]
    G.encoder.lstm.weight_hh_l0_reverse.data = mod["encoder.lstm.weight_hh_l0_reverse"]
    G.encoder.lstm.bias_ih_l0_reverse.data = mod["encoder.lstm.bias_ih_l0_reverse"]
    G.encoder.lstm.bias_hh_l0_reverse.data = mod["encoder.lstm.bias_hh_l0_reverse"]
    G.encoder.lstm.weight_ih_l1.data = mod["encoder.lstm.weight_ih_l1"]
    G.encoder.lstm.weight_hh_l1.data = mod["encoder.lstm.weight_hh_l1"]
    G.encoder.lstm.bias_ih_l1.data = mod["encoder.lstm.bias_ih_l1"]
    G.encoder.lstm.bias_hh_l1.data = mod["encoder.lstm.bias_hh_l1"]
    G.encoder.lstm.weight_ih_l1_reverse.data = mod["encoder.lstm.weight_ih_l1_reverse"]
    G.encoder.lstm.weight_hh_l1_reverse.data = mod["encoder.lstm.weight_hh_l1_reverse"]
    G.encoder.lstm.bias_ih_l1_reverse.data = mod["encoder.lstm.bias_ih_l1_reverse"]
    G.encoder.lstm.bias_hh_l1_reverse.data = mod["encoder.lstm.bias_hh_l1_reverse"]

    G.decoder.lstm1.weight_ih_l0.data = mod["decoder.lstm1.weight_ih_l0"]
    G.decoder.lstm1.weight_hh_l0.data = mod["decoder.lstm1.weight_hh_l0"]
    G.decoder.lstm1.bias_ih_l0.data   = mod["decoder.lstm1.bias_ih_l0"]
    G.decoder.lstm1.bias_hh_l0.data   = mod["decoder.lstm1.bias_hh_l0"]

    G.decoder.conv[0].weight.data = mod["decoder.convolutions.0.0.conv.weight"]
    G.decoder.conv[0].bias.data = mod["decoder.convolutions.0.0.conv.bias"]
    G.decoder.conv[1].weight.data = mod["decoder.convolutions.0.1.weight"]
    G.decoder.conv[1].bias.data = mod["decoder.convolutions.0.1.bias"]
    G.decoder.conv[1].running_mean.data = mod["decoder.convolutions.0.1.running_mean"]
    G.decoder.conv[1].running_var.data = mod["decoder.convolutions.0.1.running_var"]
    G.decoder.conv[1].num_batches_tracked.data = mod["decoder.convolutions.0.1.num_batches_tracked"]
    G.decoder.conv[3].weight.data = mod["decoder.convolutions.1.0.conv.weight"]
    G.decoder.conv[3].bias.data = mod["decoder.convolutions.1.0.conv.bias"]
    G.decoder.conv[4].weight.data = mod["decoder.convolutions.1.1.weight"]
    G.decoder.conv[4].bias.data = mod["decoder.convolutions.1.1.bias"]
    G.decoder.conv[4].running_mean.data = mod["decoder.convolutions.1.1.running_mean"]
    G.decoder.conv[4].running_var.data = mod["decoder.convolutions.1.1.running_var"]
    G.decoder.conv[4].num_batches_tracked.data = mod["decoder.convolutions.1.1.num_batches_tracked"]
    G.decoder.conv[6].weight.data = mod["decoder.convolutions.2.0.conv.weight"]
    G.decoder.conv[6].bias.data = mod["decoder.convolutions.2.0.conv.bias"]
    G.decoder.conv[7].weight.data = mod["decoder.convolutions.2.1.weight"]
    G.decoder.conv[7].bias.data = mod["decoder.convolutions.2.1.bias"]
    G.decoder.conv[7].running_mean.data = mod["decoder.convolutions.2.1.running_mean"]
    G.decoder.conv[7].running_var.data = mod["decoder.convolutions.2.1.running_var"]
    G.decoder.conv[7].num_batches_tracked.data = mod["decoder.convolutions.2.1.num_batches_tracked"]

    G.decoder.lstm2.weight_ih_l0.data = mod["decoder.lstm2.weight_ih_l0"]
    G.decoder.lstm2.weight_hh_l0.data = mod["decoder.lstm2.weight_hh_l0"]
    G.decoder.lstm2.bias_ih_l0.data   = mod["decoder.lstm2.bias_ih_l0"]
    G.decoder.lstm2.bias_hh_l0.data   = mod["decoder.lstm2.bias_hh_l0"]
    G.decoder.lstm2.weight_ih_l1.data = mod["decoder.lstm2.weight_ih_l1"]
    G.decoder.lstm2.weight_hh_l1.data = mod["decoder.lstm2.weight_hh_l1"]
    G.decoder.lstm2.bias_ih_l1.data   = mod["decoder.lstm2.bias_ih_l1"]
    G.decoder.lstm2.bias_hh_l1.data   = mod["decoder.lstm2.bias_hh_l1"]

    G.decoder.linear.weight.data = mod["decoder.linear_projection.linear_layer.weight"]
    G.decoder.linear.bias.data = mod["decoder.linear_projection.linear_layer.bias"]


    G.postnet.conv[0].weight.data               = mod["postnet.convolutions.0.0.conv.weight"]
    G.postnet.conv[0].bias.data                 = mod["postnet.convolutions.0.0.conv.bias"]
    G.postnet.conv[1].weight.data               = mod["postnet.convolutions.0.1.weight"]
    G.postnet.conv[1].bias.data                 = mod["postnet.convolutions.0.1.bias"]
    G.postnet.conv[1].running_mean.data         = mod["postnet.convolutions.0.1.running_mean"]
    G.postnet.conv[1].running_var.data          = mod["postnet.convolutions.0.1.running_var"]
    G.postnet.conv[1].num_batches_tracked.data  = mod["postnet.convolutions.0.1.num_batches_tracked"]
    G.postnet.conv[3].weight.data               = mod["postnet.convolutions.1.0.conv.weight"]
    G.postnet.conv[3].bias.data                 = mod["postnet.convolutions.1.0.conv.bias"]
    G.postnet.conv[4].weight.data               = mod["postnet.convolutions.1.1.weight"]
    G.postnet.conv[4].bias.data                 = mod["postnet.convolutions.1.1.bias"]
    G.postnet.conv[4].running_mean.data         = mod["postnet.convolutions.1.1.running_mean"]
    G.postnet.conv[4].running_var.data          = mod["postnet.convolutions.1.1.running_var"]
    G.postnet.conv[4].num_batches_tracked.data  = mod["postnet.convolutions.1.1.num_batches_tracked"]
    G.postnet.conv[5].weight.data               = mod["postnet.convolutions.2.0.conv.weight"]
    G.postnet.conv[5].bias.data                 = mod["postnet.convolutions.2.0.conv.bias"]
    G.postnet.conv[6].weight.data               = mod["postnet.convolutions.2.1.weight"]
    G.postnet.conv[6].bias.data                 = mod["postnet.convolutions.2.1.bias"]
    G.postnet.conv[6].running_mean.data         = mod["postnet.convolutions.2.1.running_mean"]
    G.postnet.conv[6].running_var.data          = mod["postnet.convolutions.2.1.running_var"]
    G.postnet.conv[6].num_batches_tracked.data  = mod["postnet.convolutions.2.1.num_batches_tracked"]
    G.postnet.conv[7].weight.data               = mod["postnet.convolutions.3.0.conv.weight"]
    G.postnet.conv[7].bias.data                 = mod["postnet.convolutions.3.0.conv.bias"]
    G.postnet.conv[8].weight.data               = mod["postnet.convolutions.3.1.weight"]
    G.postnet.conv[8].bias.data                 = mod["postnet.convolutions.3.1.bias"]
    G.postnet.conv[8].running_mean.data         = mod["postnet.convolutions.3.1.running_mean"]
    G.postnet.conv[8].running_var.data          = mod["postnet.convolutions.3.1.running_var"]
    G.postnet.conv[8].num_batches_tracked.data  = mod["postnet.convolutions.3.1.num_batches_tracked"]
    G.postnet.conv[10].weight.data              = mod["postnet.convolutions.4.0.conv.weight"]
    G.postnet.conv[10].bias.data                = mod["postnet.convolutions.4.0.conv.bias"]
    G.postnet.conv[11].weight.data              = mod["postnet.convolutions.4.1.weight"]
    G.postnet.conv[11].bias.data                = mod["postnet.convolutions.4.1.bias"]
    G.postnet.conv[11].running_mean.data        = mod["postnet.convolutions.4.1.running_mean"]
    G.postnet.conv[11].running_var.data         = mod["postnet.convolutions.4.1.running_var"]
    G.postnet.conv[11].num_batches_tracked.data = mod["postnet.convolutions.4.1.num_batches_tracked"]

torch.save(G.state_dict(), 'generator.ckpt')

G = Generator().eval().to(device)
mod = torch.load('generator.ckpt', map_location=device)
G.load_state_dict(mod)

exit()

# opt = torch.load('autovc.ckpt', map_location=device)["optimizer"]
# print(opt.keys())

# print(type(g_checkpoint['model']))

# for k in mod:
#     print(k)
# exit()

# for name, module in G.named_modules():
#     if isinstance(module, nn.Sequential):
#         continue
#     if list(module.named_children()) == []:
#         print(name, module)

# print(list(G.named_modules())[0][1])

# print([n for n, _ in G.named_modules()])
# exit()
# print(len(mod))
# print(len([n for n, _ in G.named_modules()]))
exit()
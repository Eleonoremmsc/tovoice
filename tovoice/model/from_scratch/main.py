import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras
from keras import layers, Sequential

# xavier_uniform ??? ->
# calculate_gain with self weight and linear activatio
"""
initialize the weights such that the variance of the activations
are the same across every layer
"""
#layers.Dense(input_dim = input_dim ))  #20, activation='relu', input_dim = X.shape[-1]))
#1, activation='linear')
""" is Dense same as Linear ?
 -> yes, but only without activation, so is w_init_gain really = activation?
"""




""" Linear """

def MyLinearNorm(in_dim, out_dim, bias=True, activation = 'linear'):

    model = Sequential()

    model.add(layers.Dense)        # here xavier_uniform is used

    """ is Dense same as Linear ?
     -> yes, but only without activation, so is w_init_gain really = activation?
    """


""" ConvNorm """

def MyConvNorm(self, in_channels, out_channels, kernel_size=1, stride=1,
               padding=None, dilation=1, bias=True, w_init_gain='linear'):

    model = Sequential()

    model.add(layers.Conv1D)       # xavier_uniform is applied
    # xavier_uniform = calc gain with self weight and linear activation

    if padding is None:
        assert(kernel_size % 2 == 1)
        padding = int(dilation * (kernel_size - 1) / 2)

#20, activation='relu', input_dim = X.shape[-1]))

""" Encoder """

def MyEncoder():

    # not sure what and why but this has to happen at start :

    x = x.squeeze(1).transpose(2,1) # remove all input of size 1 (dimension) and change shape to
    c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1)) # turn input into input pf size 1
    layers.Concatenate((x, c_org), dim=1)

    model = Sequential()

    # start with layers -> the function ConvNorm is called

    model.add(layers.MyConvNorm(#80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5,
                         stride=1,
                         padding=2,
                         dilation=1,
                         activation ='relu'))

    model.add(layers.BatchNormalization(512))

    # both of these * 3 and transpose at end of each i in for 3

    model.add(layers.LSTM())
    model.add(layers.Flatten())



#class MyEncoder():

def try_to_decipher_encode(dim_neck, dim_emb, freq):
    model = Sequential()
    model.add(layers.Conv1D())  #20, activation='relu', input_dim = X.shape[-1]))
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D())  #80+dim_emb if i==0 else 512,
                     #512,
                     #kernel_size=5, stride=1,
                     #padding=2,
                     #dilation=1, activation='relu')
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D())  #80+dim_emb if i==0 else 512,
                     #512,
                     #kernel_size=5, stride=1,
                     #padding=2,
                     #dilation=1, activation='relu')
    model.add(layers.LayerNormalization(axis=1))
    model.add(layers.BatchNormalization())

    model.add(layers.LSTM(512, dim_neck, 2 batch_first = True, bidirectional = True))

def forward(x, c_org):
    x = x.squeeze(1).transpose(2,1) # remove all input of size 1 (dimension) and change shape to
    c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1)) # turn input into input pf size 1
    layers.Concatenate((x, c_org), dim=1)

    # for every layer x = layer activated with "relu", and x is transposed with (1,2)

    # flatten_parameters with lstm
    model.add(layers.Flatten(x)) # x = ??
    model.add(layers.LSTM(x))

    #### what is self.dim_neck!!!!!!!

    # output of self.lstm is split with dim_neck and then concatenated , why?


""" Decoder """

def MyDecoder(self, dim_neck, dim_emb, dim_pre):

    """ Model for Decode """    # I don't know the inputs
    model = Sequential()

    """ all LSTM """
    model.add(layers.LSTM)
    model.add(layers.Flatten())
    # we will now transpose x with (1,2)

    """ all Conv with relu activation """
    model.add(layers.Conv1D)
    model.add(layers.BatchNormalization)

    model.add(layers.Conv1D)
    model.add(layers.BatchNormalization)

    model.add(layers.Conv1D)
    model.add(layers.BatchNormalization)
    # also here transpose x with (1,2)

    model.add(layers.LSTM)

    # self.linear_projection = LinearNorm(1024, 80) ??? What is LinearNorm

    # LinearNorm is the first function, it contains the linear layer
    # the output gets fed to the LinearNorm, which is printed hereunder

    model.add(layers.Dense)        # here xavier_uniform is used


""" PostNet """

def MyPostNet():
    # Five 1-d convolution with 512 channels and kernel size 5

    model = Sequential()

    """
    1.
    append model from ConvNorm function
            with (80, 512, kernel_size=5, stride=1, padding=2, dilation=1, activation ='tanh')
    also append a BatchNormalization layer
            with (512)

    2.
    for i in range (1, 5-1)
    append model from ConvNorm function
            with (512, 512, kernel_size=5, stride=1, padding=2, dilation=1, activation='tanh')
    also append a BatchNormalization layer
            with (512)

    3.
    append model from ConvNorm function
            with (512, 80, kernel_size=5, stride=1, padding=2, dilation=1, activation ='linear')
    also append a BatchNormalization layer
            with (80)

    """

def

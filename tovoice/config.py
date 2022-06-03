class Configuration:

    def __init__(self):
        self.dim_emb = 256 # Embedding dimension
        self.dim_neck = 32 # Bottleneck dimension
        self.dim_pre = 512 # TODO
        self.freq = 32 # TODO

    def init(self, **kwargs):
        self.__dict__.update(kwargs)

CFG = Configuration()

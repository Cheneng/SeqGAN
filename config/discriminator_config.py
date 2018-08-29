

class D_Config(object):
    def __init__(self):
        self.dict_size = 5000
        self.embed_dim = 64
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.feather_maps = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        self.dropout = 0.75
        self.output_class = 2


class G_Config(object):
    def __init__(self):
        self.dict_size = 5000
        self.embed_dim = 32
        self.hidden_size = 32
        self.seq_len = 20
        self.batch_first = True
        self.batch_size = 32
        self.pretrain_epoch = 120
        self.total_epoch = 20   # the total batch of the adversarial training
        self.ad_epoch = 1       # the training epoch of every adversarial train
        self.sample_num = 100
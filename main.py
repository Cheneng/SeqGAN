import torch
from train import sample_data
from generator import Generator
from discriminator import Discriminator
from config import D_Config, G_Config


REAL_DATA_PATH = './real_data.txt'
FAKE_DATA_PATH = './fake_data.txt'

g_config = G_Config()
d_config = D_Config()

# oracle model to generate the REAL DATA
oracle_model = Generator(g_config)
oracle_model.init_param(a=0, b=1)

# my model to fit the oracle model
my_model = Generator(g_config)
my_model.init_param(a=-1, b=1)  # initialize the model with a different distribution.

discriminator = Discriminator(d_config)
discriminator.init_parameters()

# To sample some data as the real data.
sample_data(model=oracle_model, save_path=REAL_DATA_PATH,
            sample_num=1000, seq_len=g_config.seq_len)



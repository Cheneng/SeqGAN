import torch.nn as nn
import torch.optim as optim
from train import sample_data, pre_train
from generator import Generator
from discriminator import Discriminator
from config import D_Config, G_Config
from dataset import SequenceDataset, ClassDataset
from torch.utils.data import DataLoader

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


g_criterion = nn.NLLLoss()

# filter the parameter not need to train
parameter = filter(lambda param: param.requires_grad, my_model.parameters())
g_optimizer = optim.Adam(parameter, lr=0.001)

g_dataset = SequenceDataset(REAL_DATA_PATH)
g_dataloader = DataLoader(g_dataset, batch_size=g_config.batch_size, shuffle=True)

pre_train(model=my_model, dataloader=g_dataloader, criterion=g_criterion,
          optimizer=g_optimizer, training_epoch=50)



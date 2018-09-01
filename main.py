import torch
import torch.nn as nn
import torch.optim as optim
from train import sample_data, train
from generator import Generator
from discriminator import Discriminator
from config import D_Config, G_Config
from dataset import SequenceDataset, ClassDataset
from torch.utils.data import DataLoader
from roll_out import Rollout


def main():
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

    # Sampling some data as the real data.
    print("Generating Real data ...")
    sample_data(model=oracle_model, save_path=REAL_DATA_PATH,
                sample_num=1000, seq_len=g_config.seq_len)

    g_criterion = nn.NLLLoss()

    # To filter the parameters not need to train
    g_parameter = filter(lambda param: param.requires_grad, my_model.parameters())
    g_optimizer = optim.Adam(g_parameter, lr=0.001)

    g_dataset = SequenceDataset(REAL_DATA_PATH)
    g_dataloader = DataLoader(g_dataset, batch_size=g_config.batch_size, shuffle=True)

    # pre-train the generator model.
    print("Pre train our Generator with MLE")
    train(model=my_model, dataloader=g_dataloader, criterion=g_criterion,
          optimizer=g_optimizer, training_epoch=1)

    # sample some data as the fake data (in order to train the discriminator)
    sample_data(model=my_model, save_path=FAKE_DATA_PATH,
                sample_num=1000, seq_len=g_config.seq_len)

    # Configuration of the Discriminator
    discriminator = Discriminator(d_config)
    discriminator.init_parameters()
    d_dataset = ClassDataset(real_path=REAL_DATA_PATH, fake_path=FAKE_DATA_PATH)
    d_dataloader = DataLoader(d_dataset, batch_size=d_config.batch_size, shuffle=True)
    d_criterion = nn.CrossEntropyLoss()
    d_parameter = filter(lambda param: param.requires_grad, discriminator.parameters())
    d_optimizer = optim.Adam(d_parameter, lr=0.001)

    # train the discriminator
    print("Pre train the Discriminator")
    train(model=discriminator, dataloader=d_dataloader, criterion=d_criterion,
          optimizer=d_optimizer, training_epoch=1)

    # Adversarial Training
    print("\nAdversarial Training...")
    gen_gan_criterion = nn.NLLLoss(size_average=False)  # not to do the sum and average the loss
    dis_gan_criterion = nn.CrossEntropyLoss(size_average=False)

    g_parameter = filter(lambda param: param.requires_grad, my_model.parameters())
    gen_gan_optim = optim.Adam(g_parameter)

    d_parameter = filter(lambda param: param.requires_grad, discriminator.parameters())
    dis_gan_optim = optim.Adam(d_parameter)

    rollout = Rollout(model=my_model, update_rate=0.8)
    for total_epoch in range(g_config.total_epoch):
        # train the generator
        for i in range(g_config.ad_epoch):
            # sample some actions. [batch_size, seq_len]
            some_data = my_model.sample(g_config.batch_size, g_config.seq_len)
            reward = rollout.get_reward(some_data, 8, discriminator=discriminator)
            print(reward)
            print(reward.size())
        # train the discriminator
        for j in range(d_config.ad_epoch):
            # re-sample some data as the negative data.
            sample_data(model=my_model, save_path=FAKE_DATA_PATH,
                        sample_num=1000, seq_len=g_config.seq_len)
            # Reload the dataset to train the Discriminator
            d_dataset = ClassDataset(real_path=REAL_DATA_PATH, fake_path=FAKE_DATA_PATH)
            d_dataloader = DataLoader(d_dataset, batch_size=d_config.batch_size, shuffle=True)
            train(model=discriminator, dataloader=d_dataloader, criterion=dis_gan_criterion,
                  training_epoch=2)


if __name__ == '__main__':
    main()



import torch
import torch.nn as nn
import torch.optim as optim
from train import sample_data, train, get_loss
from generator import Generator
from discriminator import Discriminator
from config import D_Config, G_Config
from dataset import SequenceDataset, ClassDataset
from torch.utils.data import DataLoader
from roll_out import Rollout
from helper import Visdom_line
import argparse
import visdom

parser = argparse.ArgumentParser(description="Training Configuration")
parser.add_argument("--cuda", default=None, type=int, help="Set the cuda number")
parser.add_argument("--rand_seed", default=1, type=int, help="Set the random seed number")
opt = parser.parse_args()
print(opt)

if opt.cuda is not None and not torch.cuda.is_available():
    raise ValueError("The cuda is not availiable")

torch.manual_seed(opt.rand_seed)
if opt.cuda is not None:
    torch.cuda.set_device(opt.cuda)


def main():
    vis = visdom.Visdom(env="SeqGAN")
    vis_g = Visdom_line(vis=vis, win="1", name="Generator Loss")
    vis_d = Visdom_line(vis=vis, win="2", name="Discriminator Loss")
    vis_oracle = Visdom_line(vis=vis, win="3", name="Oracle Loss")
    vis_reward = Visdom_line(vis=vis, win="4", name="Reward")

    REAL_DATA_PATH = './real_data.txt'
    FAKE_DATA_PATH = './fake_data.txt'
    EVAL_DATA_PATH = './eval_data.txt'

    g_config = G_Config()
    g_config.cuda = opt.cuda
    d_config = D_Config()

    # oracle model to generate the REAL DATA
    oracle_model = Generator(g_config)
    oracle_model.init_param(a=0, b=1)

    # my model to fit the oracle model
    my_model = Generator(g_config)
    my_model.init_param(a=-1, b=1)  # initialize the model with a different distribution.
    if opt.cuda is not None:
        my_model.cuda()
        oracle_model.cuda()

    # Sampling some data as the real data.
    print("Generating Real data ...")
    sample_data(model=oracle_model, save_path=REAL_DATA_PATH,
                sample_num=g_config.sample_num, seq_len=g_config.seq_len)

    g_criterion = nn.NLLLoss()

    # To filter the parameters not need to train
    g_parameter = filter(lambda param: param.requires_grad, my_model.parameters())
    g_optimizer = optim.Adam(g_parameter, lr=0.001)

    g_dataset = SequenceDataset(REAL_DATA_PATH)
    g_dataloader = DataLoader(g_dataset, batch_size=g_config.batch_size, shuffle=True,
                              pin_memory=True if opt.cuda is not None else False)

    # pre-train the generator model.
    print("Pre train our Generator with MLE")
    for i in range(g_config.pretrain_epoch):
        train(model=my_model, dataloader=g_dataloader, criterion=g_criterion, opt=opt,
              optimizer=g_optimizer, training_epoch=1, name="Pre-train the Generator", vis=vis_g)
        # after each epoch pre-train, we sample some evaluate data compare with the oracle model
        sample_data(model=my_model, save_path=EVAL_DATA_PATH,
                    sample_num=g_config.sample_num, seq_len=g_config.seq_len)
        eval_dataset = SequenceDataset(EVAL_DATA_PATH)
        eval_dataloader = DataLoader(eval_dataset, batch_size=100, pin_memory=True if opt.cuda is not None else False)
        get_loss(model=oracle_model, dataloader=eval_dataloader, criterion=g_criterion, vis=vis_oracle)

    # sample some data as the fake data (in order to train the discriminator)
    sample_data(model=my_model, save_path=FAKE_DATA_PATH,
                sample_num=g_config.sample_num, seq_len=g_config.seq_len)

    # Configuration of the Discriminator
    discriminator = Discriminator(d_config)
    discriminator.init_parameters()

    if opt.cuda is not None:
        discriminator.cuda()

    d_dataset = ClassDataset(real_path=REAL_DATA_PATH, fake_path=FAKE_DATA_PATH)
    d_dataloader = DataLoader(d_dataset, batch_size=d_config.batch_size, shuffle=True,
                              pin_memory=True if opt.cuda is not None else False)
    d_criterion = nn.CrossEntropyLoss()
    d_parameter = filter(lambda param: param.requires_grad, discriminator.parameters())
    d_optimizer = optim.Adam(d_parameter, lr=0.001)

    # train the discriminator
    print("Pre train the Discriminator")
    train(model=discriminator, dataloader=d_dataloader, criterion=d_criterion, opt=opt,
          optimizer=d_optimizer, training_epoch=d_config.pretrain_epoch, name="pre-train the Discriminator", vis=vis_d)
    discriminator.eval()

    # Adversarial Training
    print("\nAdversarial Training...")
    gen_gan_criterion = nn.NLLLoss()
    dis_gan_criterion = nn.CrossEntropyLoss(size_average=False)

    g_parameter = filter(lambda param: param.requires_grad, my_model.parameters())
    gen_gan_optim = optim.Adam(g_parameter, lr=0.01)

    d_parameter = filter(lambda param: param.requires_grad, discriminator.parameters())
    dis_gan_optim = optim.Adagrad(d_parameter, lr=0.001)

    rollout = Rollout(model=my_model, update_rate=0.8)
    for total_epoch in range(g_config.total_epoch):
        print("Training the Generator ... ")
        for i in range(g_config.ad_epoch):
            # sample some actions. [batch_size * seq_len]
            some_data = my_model.sample(g_config.batch_size, g_config.seq_len)
            reward = rollout.get_reward(some_data, g_config.num_rollout, discriminator=discriminator)
            reward = reward.detach()   # don't need to compute the gradient of the reward.

            # draw the reward line
            vis_reward.update(torch.sum(reward).item() / (g_config.batch_size * g_config.seq_len))

            # To get the sample rate to optim.
            init_zero_input = torch.zeros((some_data.size(0), 1)).long()
            if opt.cuda is not None:
                init_zero_input = init_zero_input.cuda()
            input_data = torch.cat([init_zero_input, some_data[:, :-1].contiguous()], dim=1)
            output_rate = my_model(input_data)
            policy_output_rate = (reward.unsqueeze(-1) * output_rate)
            loss = gen_gan_criterion(policy_output_rate.view(-1, policy_output_rate.size(-1)), some_data.view(-1))

            # print(loss.item())
            gen_gan_optim.zero_grad()
            loss.backward()
            gen_gan_optim.step()

            # sample some generated data to evaluate training
            sample_data(model=my_model, save_path=EVAL_DATA_PATH,
                        sample_num=g_config.sample_num, seq_len=g_config.seq_len)
            eval_dataset = SequenceDataset(EVAL_DATA_PATH)
            eval_dataloader = DataLoader(eval_dataset, batch_size=100, pin_memory=True if opt.cuda is not None else False)
            get_loss(model=oracle_model, dataloader=eval_dataloader, criterion=g_criterion, vis=vis_oracle)

            # update the parameters of the roll-out model
            rollout.update_param()

        # train the discriminator
        for j in range(d_config.ad_epoch):
            discriminator.train()
            # re-sample some data as the negative data. (After update the parameters)
            sample_data(model=my_model, save_path=FAKE_DATA_PATH,
                        sample_num=g_config.sample_num, seq_len=g_config.seq_len)
            # Reload the dataset to train the Discriminator
            d_dataset = ClassDataset(real_path=REAL_DATA_PATH, fake_path=FAKE_DATA_PATH)
            d_dataloader = DataLoader(d_dataset, batch_size=d_config.batch_size, shuffle=True,
                                      pin_memory=True if opt.cuda is not None else False)
            train(model=discriminator, dataloader=d_dataloader, criterion=dis_gan_criterion, opt=opt,
                  optimizer=dis_gan_optim, training_epoch=2, name="Discriminator", vis=vis_d)
            discriminator.eval()


if __name__ == '__main__':
    main()



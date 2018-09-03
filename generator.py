import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Generator(nn.Module):
    def __init__(self, config):
        """
        :param config: G_Config object Include the configuration of this Module.
        """
        super(Generator, self).__init__()
        self.input_size = config.embed_dim
        self.hidden_size = config.hidden_size
        self.output_size = config.dict_size
        self.embed = nn.Embedding(config.dict_size, config.embed_dim)
        self.embed.weight.requires_grad = False     # not to change the embedding
        self.lstm = nn.LSTM(self.input_size, self.hidden_size,
                            batch_first=config.batch_first)
        self.linear_out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax_out = nn.LogSoftmax(dim=2)
        self.use_cuda = True if config.cuda is not None else False

    def forward(self, x, hidden=None):
        """
        :param x: The input data (dim=2)
        :return: the output of the sequence.
        """
        x = self.embed(x)
        self.lstm.flatten_parameters()
        x, hidden = self.lstm(x, hidden)
        x = self.linear_out(x)
        x = self.softmax_out(x)
        return x

    def step_forward_sample(self, x, hidden=None):
        """
            Run the model one step.
        :param x: (torch.LongTensor) The input of the step.
        :param hidden: (torch.FloatTensor) The hidden input of the step.
        :return: (output max index, hidden)
        """
        if self.use_cuda:
            x = x.cuda()
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear_out(x)
        x = F.softmax(x, dim=2).squeeze(1)
        x = x.multinomial(1)
        return x, hidden

    def sample(self, batch_size, seq_len):
        """
            Sample some data from the model from the beginning.
        :param batch_size: the batch of the sample data.
        :param seq_len: the sequence length of the sample data.
        :return: (torch.LongTensor) the sample data indexes whose first input index is 0.
        """
        output = []
        x = torch.zeros((batch_size, 1)).long()
        if self.use_cuda:
            x = x.cuda()
        hidden = self.init_hidden(batch_size)

        for _ in range(seq_len):
            x, hidden = self.step_forward_sample(x, hidden)
            output.append(x)
        output = torch.cat(output, dim=1)
        return output

    def partial_sample(self, seq_len, partial_data):
        batch_size = partial_data.size(0)
        given_len = partial_data.size(1)

        # not need to generate
        if given_len == seq_len:
            return partial_data

        # if actually not the partial sample ...
        if given_len == 0:
            return sample_data(batch_size=batch_size, seq_len=seq_len)

        first_input = torch.zeros((batch_size, 1)).long()
        if self.use_cuda:
            first_input = first_input.cuda()
        input_ = torch.cat([first_input, partial_data], dim=1)
        input_ = input_.contiguous()

        hidden = self.init_hidden(batch_size)
        hidden = hidden
        # encode to get the hidden
        input_ = self.embed(input_)
        # input = input.contiguous()
        # self.lstm.flatten_parameters()
        out, hidden = self.lstm(input_, hidden)
        out = self.linear_out(out)
        x = F.softmax(out[:, -1, :], dim=1)
        x = x.multinomial(1)     # as the next step input

        out_list = []
        for i in range(seq_len - given_len):
            out_list.append(x)
            x, hidden = self.step_forward_sample(x, hidden)
        out = torch.cat(out_list, dim=1)
        out = torch.cat([partial_data, out], dim=1)
        return out

    def init_hidden(self, batch_size):
        h = torch.zeros((1, batch_size, self.hidden_size))
        c = torch.zeros((1, batch_size, self.hidden_size))
        if self.use_cuda:
            h, c = h.cuda(), c.cuda()
        return h, c

    def init_param(self, a=0, b=1):
        """
            To initialize the model parameter with normal distribution (a, b).
        :param a: the expectation of the normal distribution.
        :param b: the variance of the normal distribution.
        """
        for param in self.parameters():
            param.data.normal_(a, b)


if __name__ == '__main__':
    from config import G_Config
    from train import sample_data
    config = G_Config()
    model = Generator(config)
    model.init_param()
    out = model.sample(batch_size=20, seq_len=10)
    sample_data(model)
    print(out)
    pred = model.forward(out)
    print(pred.size())


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
        self.softmax_out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        :param x: The input data (dim=2)
        :return: the output of the sequence.
        """
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.softmax_out(x)
        return x, hidden

    def step_forward_sample(self, x, hidden=None):
        """
            Run the model one step.
        :param x: (torch.LongTensor) The input of the step.
        :param hidden: (torch.FloatTensor) The hidden input of the step.
        :return: (output max index, hidden)
        """
        x = self.embed(x)
        x, hidden = self.lstm(x, hidden)
        x = F.softmax(x, dim=2).squeeze(1)
        x = x.multinomial(1)
        return x, hidden

    def sample(self, batch_size, seq_len):
        """
            Sample some data from the model.
        :param batch_size: the batch of the sample data.
        :param seq_len: the sequence length of the sample data.
        :return: (torch.LongTensor) the sample data indexes whose first input index is 0.
        """
        output = []
        x = torch.zeros((batch_size, 1)).long()
        if torch.cuda.is_available() is True:
            x = x.cuda()

        hidden = self.init_hidden(batch_size)
        for _ in range(seq_len):
            x, hidden = self.step_forward_sample(x, hidden)
            output.append(x)
        output = torch.cat(output, dim=1)
        return output

    def init_hidden(self, batch_size):
        h = torch.zeros((1, batch_size, self.hidden_size))
        c = torch.zeros((1, batch_size, self.hidden_size))
        if torch.cuda.is_available():
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
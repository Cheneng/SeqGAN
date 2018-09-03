import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    TextCNN for classification.
    """
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.dict_size, config.embed_dim)
        self.embed.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, config.embed_dim)) for (n, f) in zip(config.feather_maps, config.filter_sizes)
        ])
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.linear_1 = nn.Linear(sum(config.feather_maps), sum(config.feather_maps))
        self.linear_out = nn.Linear(sum(config.feather_maps), config.output_class)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len] LongTensor
        :return: the output class rate
        """
        x = self.embed(x)
        x = x.unsqueeze(1)  # [batch_size, (input_channel : 1), seq_len, embedding]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        x = [F.max_pool1d(data, data.size(2)).squeeze(2) for data in x]
        x = torch.cat(x, dim=-1)

        temp = self.linear_1(x)
        gate = F.sigmoid(temp)
        x = gate * F.relu(temp) + (1.0 - gate) * x

        x = self.dropout(x)
        x = self.linear_out(x)
        return x

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


if __name__ == '__main__':
    from config import D_Config
    config = D_Config()
    model = Discriminator(config)
    input_data = torch.randint(0, 10, size=(10, 20)).long()
    out = model(input_data)
    print(out)

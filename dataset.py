import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.read_data_file(data_path)

    @staticmethod
    def read_data_file(data_path):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                temp_list = list(map(int, line.strip().split(' ')))
                data.append(temp_list)
        return data

    def __getitem__(self, index):
        label = self.data[index]
        data = [0]
        data.extend(label[:-1])
        return torch.Tensor(data).long(), torch.Tensor(label).long()

    def __len__(self):
        return len(self.data)


class ClassDataset(Dataset):
    def __init__(self, real_path, fake_path):
        real_data = self.read_data_file(real_path)
        fake_data = self.read_data_file(fake_path)
        self.data = real_data + fake_data
        self.target = [1 for _ in range(len(real_data))] + [0 for _ in range(len(fake_data))]

    @staticmethod
    def read_data_file(data_path):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                temp_list = list(map(int, line.strip().split(' ')))
                data.append(temp_list)
        return data

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), torch.Tensor([self.target[index]]).long()

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data = ClassDataset('./real_data.txt', './fake_data.txt')
    trainloader = DataLoader(dataset=data, batch_size=2, shuffle=False)

    for index, (x, y) in enumerate(trainloader):
        print(index)
        print(x)


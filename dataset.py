import torch
from torch.utils.data import dataset


class SequenceDataset(dataset):
    def __init__(self, data_path):
        self.data = self._read_data_file(data_path)


    def _read_data_file(self, data_path):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                temp_list = list(map(int, line.strip().split(' ')))
                data.append(temp_list)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ClassDataset(dataset):
    def __init__(self, real_path, fake_path):
        self.data =

    def _read_data_file(self, data_path):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                temp_list = list(map(int, line.strip().split(' ')))
                data.append(temp_list)
        return data
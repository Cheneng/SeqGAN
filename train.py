import torch
import torch.nn as nn

def pre_train(model, dataloader, optimizer, training_epoch, batch_size):
    for epoch in range(training_epoch):
        for data in dataloader:
            



def sample_data(model, save_path='./real_data.txt',
                sample_num=1000, batch_size=100, seq_len=20):
    with open(save_path, 'w') as f:
        for i in range(sample_num // batch_size):
            out = model.sample(batch_size, seq_len).tolist()
            for sample in out:
                f.write("%s\n" % ' '.join(map(str, sample)))





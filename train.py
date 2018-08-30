import torch
import torch.nn as nn


def pre_train(model, dataloader, criterion, optimizer,
              training_epoch=20, vis=None):
    for epoch in range(training_epoch):
        for step, (data, label) in enumerate(dataloader):
            out = model(data)
            out = out.view(-1, out.size(-1))
            label = label.view(-1)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss.item())

            # TODO using visdom to visualize the loss curve.
            if vis is not None:
                pass


def sample_data(model, save_path='./real_data.txt',
                sample_num=1000, batch_size=100, seq_len=20):
    with open(save_path, 'w') as f:
        for i in range(sample_num // batch_size):
            out = model.sample(batch_size, seq_len).tolist()
            for sample in out:
                f.write("%s\n" % ' '.join(map(str, sample)))





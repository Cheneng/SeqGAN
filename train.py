import torch
import torch.nn as nn


def train(model, dataloader, criterion, optimizer, opt,
          training_epoch=20, vis=None, name='None'):
    for epoch in range(training_epoch):
        loss_epoch = 0
        step = 0
        for step, (data, label) in enumerate(dataloader, 1):
            if opt.cuda is not None:
                data = data.cuda()
                label = label.cuda()
            out = model(data)
            out = out.view(-1, out.size(-1))
            label = label.view(-1)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        if vis is not None:
            vis.update(loss_epoch/step)

        print("{0} Loss {1}".format(name, loss_epoch/step))


def get_loss(model, dataloader, criterion, vis=None):
    loss_epoch = 0
    step = 0
    for step, (data, label) in enumerate(dataloader, 1):
        if model.use_cuda:
            data = data.cuda()
            label = label.cuda()
        out = model(data)
        out = out.view(-1, out.size(-1))
        label = label.view(-1)
        loss = criterion(out, label)
        loss_epoch += loss.item()

    if vis is not None:
        vis.update(loss_epoch/step)
    return loss_epoch


def sample_data(model, save_path='./real_data.txt',
                sample_num=1000, batch_size=100, seq_len=20):
    with open(save_path, 'w') as f:
        for i in range(sample_num // batch_size):
            out = model.sample(batch_size, seq_len).tolist()
            for sample in out:
                f.write("%s\n" % ' '.join(map(str, sample)))




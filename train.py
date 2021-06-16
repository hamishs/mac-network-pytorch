import sys
import pickle
from collections import Counter

import wandb
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning import seed_everything

from dataset import CLEVR, collate_data, transform
from model import MACNetwork


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accumulate(model1, model2, decay=0.999):
    for p, q in zip(model1.parameters(), model2.parameters()):
        p.data += (1.0 - decay) * (q.data - p.data)


def train(dataset_root, net, net_running, criterion, optimizer, epoch, batch_size,
          log_wandb=False):
    clevr = CLEVR(dataset_root, transform=transform)
    train_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data
    )

    dataset = iter(train_set)
    pbar = tqdm(dataset)
    moving_loss = 0

    net.train(True)
    for image, question, q_len, answer, _ in pbar:
        image, question, answer = (
            image.to(device),
            question.to(device),
            answer.to(device),
        )

        net.zero_grad()
        output = net(image, question, q_len)
        loss = criterion(output, answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == answer
        correct = correct.float().sum() / batch_size

        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Loss: {:.5f}; Acc: {:.5f}'.format(
                epoch + 1, loss.item(), moving_loss
            )
        )
        if log_wandb:
            wandb.log({'loss':loss.item(), 'acc':moving_loss})

        accumulate(net_running, net)

    clevr.close()


def valid(dataset_root, net, epoch, batch_size, log_wandb=False):
    clevr = CLEVR(dataset_root, 'val', transform=None)
    valid_set = DataLoader(
        clevr, batch_size=batch_size, num_workers=4, collate_fn=collate_data
    )
    dataset = iter(valid_set)

    net.train(False)
    family_correct = Counter()
    family_total = Counter()
    with torch.no_grad():
        for image, question, q_len, answer, family in tqdm(dataset):
            image, question = image.to(device), question.to(device)

            output = net(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    with open('log/log_{}.txt'.format(str(epoch + 1).zfill(2)), 'w') as w:
        for k, v in family_total.items():
            w.write('{}: {:.5f}\n'.format(k, family_correct[k] / v))

    print(
        'Avg Acc: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values())
        )
    )

    if log_wandb:
        wandb.log({'val_acc':sum(family_correct.values()) / sum(family_total.values())})

    clevr.close()


def main(params):
    seed_everything(params.seed, workers=True)

    if params.wandb != 'none':
        wandb.init(
            project='Sparse-VQA',
            entity=params.wandb,
            group='MAC',
            config={k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str, list))}
        )
    
    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)

    n_words = len(dic['word_dic']) + 1
    n_answers = len(dic['answer_dic'])

    net = MACNetwork(n_words, params.dim, embed_hidden=params.embed_hidden,
        max_step=params.max_step, self_attention=params.self_attention,
        memory_gate=params.memory_gate, classes=params.classes,
        dropout=params.dropout, activation=params.activation).to(device)
    net_running = MACNetwork(n_words, params.dim, embed_hidden=params.embed_hidden,
        max_step=params.max_step, self_attention=params.self_attention,
        memory_gate=params.memory_gate, classes=params.classes,
        dropout=params.dropout, activation=params.activation).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params.lr)

    for epoch in range(params.n_epoch):
        train(params.dataset_root, net, net_running, criterion, optimizer,
              epoch, params.batch_size, log_wandb=(params.wandb!='none'))
        valid(params.dataset_root, net_running,
              epoch, params.batch_size, log_wandb=(params.wandb!='none'))

        with open(
            'checkpoint/checkpoint_{}.model'.format(params.name), 'wb'
        ) as f:
            torch.save(net_running.state_dict(), f)
    
    wandb.finish()


if __name__ == '__main__':
    from loader import Loader 
    from arguments import parser
    params = Loader(parser.parse_args())
    main(params)
    

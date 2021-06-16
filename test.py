import sys
import pickle
from collections import Counter

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CLEVR, collate_data, transform
from model import MACNetwork


def main(params):

    test_set = DataLoader(
        CLEVR(sys.argv[1], 'val', transform=None),
        batch_size=params.batch_size,
        num_workers=4,
        collate_fn=collate_data,
    )

    with open('data/dic.pkl', 'rb') as f:
        dic = pickle.load(f)
    n_words = len(dic['word_dic']) + 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MACNetwork(n_words, params.dim, embed_hidden=params.embed_hidden,
            max_step=params.max_step, self_attention=params.self_attention,
            memory_gate=params.memory_gate, classes=params.classes,
            dropout=params.dropout, activation=params.activation).to(device)
    net.load_state_dict(torch.load('checkpoint/checkpoint_{}.model'.format(params.name)))
    net.eval()

    dataset = iter(test_set)
    pbar = tqdm(dataset)
    family_correct = Counter()
    family_total = Counter()

    for image, question, q_len, answer, family in pbar:
        image, question = image.to(device), question.to(device)
        with torch.no_grad():
            output = net(image, question, q_len)
        correct = output.argmax(1) == answer.to(device)
        for c, fam in zip(correct, family):
            if c:
                family_correct[fam] += 1
            family_total[fam] += 1

    print(
        'Avg Acc: {:.5f}'.format(
            sum(family_correct.values()) / sum(family_total.values())
        )
    )

    with open('log/log_test.txt', 'w') as w:
        w.write(
            'Avg Acc: {:.5f}'.format(sum(family_correct.values()) / sum(family_total.values()))
        )


if __name__ == '__main__':
    from loader import load_params
    from arguments import parser
    params = load_params('checkpoint/params_{}.json'.format(parser.parse_args().name))
    main(params)


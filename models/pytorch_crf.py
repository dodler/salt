import random

import torch
from torchcrf import CRF

DEVICE = 0

crf = CRF(5)


def make_emissions(seq_length=3, batch_size=2, num_tags=5):
    return torch.randn(seq_length, batch_size, num_tags)


def make_tags(seq_length=3, batch_size=2, num_tags=5):
    return torch.LongTensor([
        [random.randrange(num_tags) for b in range(batch_size)]
        for _ in range(seq_length)
    ])


emissions = make_emissions()
print(emissions.size(), emissions)
tags = make_tags()
print(tags.size(), tags)
print(crf(emissions, tags))

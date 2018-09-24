import unittest
import torch

from models.decoders import DecoderBlockV2


class TestDecoderBlocks(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_block_input(self):
        inp = torch.zeros((128,12, 28,28))
        print(inp.size())
        dec = DecoderBlockV2(12,2,2)
        print(dec(inp))
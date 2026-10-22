"""Exercise initial zero input through the real cerebellum FIR blocks."""

from contextlib import chdir
from pathlib import Path
import sys
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
from diva_cerebellum import Cerebellum


class CerebellumInitializationTests(unittest.TestCase):
    def test_two_second_target_has_valid_first_output(self):
        with chdir(SOURCE):
            block = Cerebellum(401)
            block.input(torch.zeros((401, 1), dtype=torch.float64))
            block.output()
        self.assertEqual(tuple(block.OutputPorts[0].shape), (802, 1))
        torch.testing.assert_close(block.OutputPorts[0], torch.zeros((802, 1), dtype=torch.float64), rtol=0, atol=0)

    def test_short_target_keeps_legacy_initialization(self):
        with chdir(SOURCE):
            block = Cerebellum(61)
            self.assertEqual(tuple(block.dataBuffer[0].shape), (101, 1))
            block.input(torch.zeros((61, 1), dtype=torch.float64))
            block.output()
        torch.testing.assert_close(block.OutputPorts[0], torch.zeros((122, 1), dtype=torch.float64), rtol=0, atol=0)

    def test_long_initialization_matches_explicit_zero_padding(self):
        with chdir(SOURCE):
            actual, reference = Cerebellum(401), Cerebellum(401)
            reference.dataBuffer[0] = torch.zeros((401, 1), dtype=torch.float64)
            for frame in range(16):
                signal = torch.zeros((401, 1), dtype=torch.float64)
                signal[frame] = 1
                for block in (actual, reference):
                    block.input(signal)
                    block.output()
                torch.testing.assert_close(actual.OutputPorts[0], reference.OutputPorts[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

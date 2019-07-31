import unittest
from minari.trainer import Trainer
from minari.loss import RMSELoss


class TestTrainer(unittest.TestCase):
    def test_default_constructor(self):
        trainer = Trainer(loss=RMSELoss())

        self.assertEqual(64, trainer.batch_size)

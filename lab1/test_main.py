"""Небольшая проверка сети и разделения локальных данных."""

import unittest

import torch
from torch import nn

from main import CNN, load_data


class LabTests(unittest.TestCase):
    def test_model_and_gradients(self):
        model = CNN()
        outputs = model(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(outputs.shape), (4, 10))
        loss = nn.CrossEntropyLoss()(outputs, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_local_dataset_and_split(self):
        train, validation, test = load_data()
        self.assertEqual((len(train), len(validation), len(test)), (54000, 6000, 10000))
        self.assertFalse(set(train.indices) & set(validation.indices))
        image, label = train[0]
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertTrue(0 <= image.min() <= image.max() <= 1)
        self.assertTrue(0 <= label.item() < 10)


if __name__ == "__main__":
    unittest.main()

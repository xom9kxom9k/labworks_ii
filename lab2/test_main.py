"""Проверки аугментации, разбиения данных и обучения сети."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import AugmentedDataset, CNN, load_data, run_epoch


class LabTests(unittest.TestCase):
    def test_augmentation_preserves_source_and_labels(self):
        image = torch.zeros(1, 1, 28, 28)
        image[:, :, 5:20, 8:12] = 1
        source = TensorDataset(image, torch.tensor([6]))
        augmented = AugmentedDataset(source)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            outputs = [augmented[0] for _ in range(10)]
        self.assertEqual(len(augmented), len(source))
        self.assertTrue(all(label.item() == 6 for _, label in outputs))
        self.assertTrue(all(tuple(x.shape) == (1, 28, 28) for x, _ in outputs))
        self.assertTrue(all(torch.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1
                            for x, _ in outputs))
        self.assertTrue(any(not torch.equal(x, image[0]) for x, _ in outputs))
        expected = torch.zeros_like(image)
        expected[:, :, 5:20, 8:12] = 1
        self.assertTrue(torch.equal(image, expected))
        augmented.enabled = False
        self.assertTrue(torch.equal(augmented[0][0], image[0]))

    def test_split_and_unchanged_evaluation_data(self):
        train, validation, test = load_data()
        self.assertEqual((len(train), len(validation), len(test)), (54000, 6000, 10000))
        self.assertFalse(set(train.indices) & set(validation.indices))
        augmented = AugmentedDataset(train)
        self.assertIs(augmented.source, train)
        before = validation[0][0].clone(), test[0][0].clone()
        augmented[0]
        self.assertTrue(torch.equal(before[0], validation[0][0]))
        self.assertTrue(torch.equal(before[1], test[0][0]))

    def test_training_and_evaluation(self):
        torch.set_num_threads(2)
        model = CNN()
        source = TensorDataset(torch.rand(8, 1, 28, 28), torch.arange(8))
        loader = DataLoader(AugmentedDataset(source), batch_size=4)
        before = model.layers[0].weight.detach().clone()
        metrics = run_epoch(model, loader, nn.CrossEntropyLoss(),
                            torch.optim.Adam(model.parameters(), lr=0.001))
        self.assertFalse(torch.equal(before, model.layers[0].weight))
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertGreater(metrics['loss'], 0)
        run_epoch(model, DataLoader(source, batch_size=4), nn.CrossEntropyLoss())
        self.assertFalse(model.training)
        self.assertEqual(tuple(model(source.tensors[0]).shape), (8, 10))


if __name__ == "__main__":
    unittest.main()

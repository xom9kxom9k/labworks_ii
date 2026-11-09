"""Проверки сдвига целевых символов, обучения и генерации."""
import unittest
import torch
from main import TextGRU, make_windows, generate, run_epoch
from torch.utils.data import DataLoader


class TextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_windows(self):
        data = make_windows(torch.arange(21), length=5)
        self.assertEqual(len(data), 4)
        for x, y in data:
            self.assertTrue(torch.equal(x + 1, y))
        self.assertEqual(data[-1][1][-1].item(), 20)
        with self.assertRaises(ValueError):
            make_windows(torch.arange(5))

    def test_learning(self):
        model = TextGRU(3)
        data = make_windows(torch.tensor([0, 1, 2] * 101), length=15)
        loader = DataLoader(data, batch_size=20)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        before = run_epoch(model, loader)
        for _ in range(4):
            run_epoch(model, loader, optimizer)
        self.assertLess(run_epoch(model, loader), before)
        logits, hidden = model(torch.zeros(2, 7, dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 7, 3))
        self.assertEqual(tuple(hidden.shape), (1, 2, 256))

    def test_generation(self):
        model = TextGRU(3)
        args = (model, ['a', 'b', '\n'], 'ab', 30, .8)
        text = generate(*args)
        self.assertEqual(text, generate(*args))
        self.assertTrue(text.startswith('ab'))
        self.assertEqual(len(text), 32)
        self.assertLessEqual(set(text), set('ab\n'))
        for prompt, temperature in [('', .8), ('x', .8), ('a', 0), ('a', float('nan'))]:
            with self.assertRaises(ValueError):
                generate(model, ['a', 'b', '\n'], prompt, temperature=temperature)


if __name__ == '__main__':
    unittest.main()

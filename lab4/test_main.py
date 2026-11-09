"""Проверки данных, метрик и обучения сегментационной сети."""
import unittest
import torch
from main import UNet, Pedestrians, scores, segmentation_loss, load_data


class SegmentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_metrics(self):
        result = scores(3, 1, 2, 4)
        self.assertAlmostEqual(result['iou'], .5)
        self.assertAlmostEqual(result['dice'], 2 / 3)
        self.assertAlmostEqual(result['pixel_accuracy'], .7)
        self.assertEqual(scores(0, 0, 0, 10)['iou'], 1)
        self.assertEqual(scores(0, 0, 10, 0)['dice'], 0)

    def test_learning(self):
        model = UNet()
        image = torch.rand(2, 3, 32, 32)
        mask = (image[:, :1] > .5).float()
        before = model.output.weight.detach().clone()
        logits = model(image)
        self.assertEqual(logits.shape, mask.shape)
        loss = segmentation_loss(logits, mask)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.output.weight))
        self.assertLess(segmentation_loss(torch.where(mask.bool(), 20., -20.), mask), .0001)

    def test_paired_flip(self):
        masks = torch.arange(16).reshape(1, 1, 4, 4).float()
        images = masks.repeat(1, 3, 1, 1)
        dataset = Pedestrians(images, masks, [0], augment=True)
        changed = False
        for _ in range(10):
            image, mask = dataset[0]
            self.assertTrue(torch.equal(image[:1], mask))
            changed |= not torch.equal(mask, masks[0])
        self.assertTrue(changed)
        self.assertTrue(torch.equal(masks.flatten(), torch.arange(16).float()))

    def test_data(self):
        parts, metadata = load_data()
        names = [set(metadata['split'][key]) for key in ['train', 'validation', 'test']]
        self.assertEqual([len(p) for p in parts], [120, 25, 25])
        self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
        self.assertEqual(len(set.union(*names)), 170)
        image, mask = parts[1][0]
        self.assertEqual(tuple(image.shape), (3, 128, 128))
        self.assertEqual(tuple(mask.shape), (1, 128, 128))
        self.assertLessEqual(set(mask.unique().tolist()), {0., 1.})
        self.assertTrue(torch.equal(parts[1][0][0], image))
        self.assertTrue(mask.any())


if __name__ == '__main__':
    unittest.main()

"""Бинарная сегментация пешеходов на Penn-Fudan с небольшой U-Net."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
SIZE = 128


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = block(3, 16)
        self.encoder2 = block(16, 32)
        self.bottom = block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder2 = block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.decoder1 = block(32, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, images):
        first = self.encoder1(images)
        second = self.encoder2(self.pool(first))
        features = self.bottom(self.pool(second))
        features = self.decoder2(torch.cat([self.up2(features), second], dim=1))
        features = self.decoder1(torch.cat([self.up1(features), first], dim=1))
        return self.output(features)


class Pedestrians(Dataset):
    def __init__(self, images, masks, indices, augment=False):
        self.images, self.masks = images, masks
        self.indices, self.augment = list(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        image, mask = self.images[source], self.masks[source]
        if self.augment and torch.rand(()) < .5:
            image, mask = image.flip(-1), mask.flip(-1)
        return image, mask


def load_data():
    folder = ROOT / 'data/PennFudanPed'
    files = sorted((folder / 'PNGImages').glob('*.png'))
    if len(files) != 170:
        raise SystemExit('Нет полного датасета. Выполните python prepare_data.py')
    images, masks = [], []
    digest = hashlib.sha256()
    for path in files:
        mask_path = folder / 'PedMasks' / (path.stem + '_mask.png')
        if not mask_path.exists():
            raise SystemExit(f'Нет маски для {path.name}')
        for item in (path, mask_path):
            digest.update(item.name.encode())
            digest.update(item.read_bytes())
        with Image.open(path) as image:
            rgb = image.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.array(rgb).copy()).permute(2, 0, 1).float() / 255)
        with Image.open(mask_path) as mask:
            labels = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
            masks.append(torch.from_numpy((np.array(labels) > 0).copy()).unsqueeze(0).float())
    order = torch.randperm(len(files), generator=torch.Generator().manual_seed(SEED)).tolist()
    groups = [order[:120], order[120:145], order[145:]]
    images, masks = torch.stack(images), torch.stack(masks)
    datasets = [Pedestrians(images, masks, indices, augment=i == 0) for i, indices in enumerate(groups)]
    metadata = {'seed': SEED, 'size': SIZE, 'sha256': digest.hexdigest(),
                'split': {key: [files[i].name for i in indices] for key, indices in
                          zip(['train', 'validation', 'test'], groups)}}
    return datasets, metadata


def segmentation_loss(logits, masks):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
    probabilities = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (probabilities * masks).sum(dims) + 1) / (probabilities.sum(dims) + masks.sum(dims) + 1)
    return bce + (1 - dice).mean()


def scores(tp, fp, fn, tn):
    return {'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            'dice': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            'pixel_accuracy': (tp + tn) / (tp + fp + fn + tn)}


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = count = tp = fp = fn = tn = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, masks in loader:
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            prediction, truth = logits.detach() >= 0, masks.bool()
            tp += (prediction & truth).sum().item()
            fp += (prediction & ~truth).sum().item()
            fn += (~prediction & truth).sum().item()
            tn += (~prediction & ~truth).sum().item()
            loss_sum += loss.item() * len(images)
            count += len(images)
    return {'loss': loss_sum / count, **scores(tp, fp, fn, tn),
            'pixels': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


@torch.no_grad()
def evaluate_examples(model, dataset, names, output):
    model.eval()
    rows = []
    predictions = []
    for index, (image, mask) in enumerate(dataset):
        prediction = model(image.unsqueeze(0))[0, 0].sigmoid() >= .5
        truth = mask[0].bool()
        values = scores((prediction & truth).sum().item(), (prediction & ~truth).sum().item(),
                        (~prediction & truth).sum().item(), (~prediction & ~truth).sum().item())
        rows.append({'file': names[index], **values})
        predictions.append(prediction)
    save_json(output / 'per_image.json', rows)
    def plot(indices, filename, title):
        fig, axes = plt.subplots(len(indices), 4, figsize=(10, 2.45 * len(indices)), squeeze=False)
        for row, index in enumerate(indices):
            image, mask = dataset[index]
            rgb = image.permute(1, 2, 0).numpy()
            prediction = predictions[index].numpy()
            axes[row, 0].imshow(rgb)
            axes[row, 1].imshow(mask[0], cmap='gray', vmin=0, vmax=1)
            axes[row, 2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
            axes[row, 3].imshow(rgb)
            overlay = np.zeros((SIZE, SIZE, 4))
            overlay[prediction] = [1, 0, 0, .45]
            axes[row, 3].imshow(overlay)
            axes[row, 0].set_ylabel(names[index], fontsize=8)
            axes[row, 2].set_title(f"IoU = {rows[index]['iou']:.3f}", fontsize=9)
            for axis in axes[row]:
                axis.set(xticks=[], yticks=[])
        for axis, label in zip(axes[0], ['Изображение', 'Истинная маска', 'Предсказанная маска', 'Наложение']):
            axis.set_title(label, fontsize=10)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
    plot(list(range(4)), 'predictions.png', 'Первые четыре изображения тестовой выборки')
    worst = sorted(range(len(rows)), key=lambda i: rows[i]['iou'])[:4]
    plot(worst, 'errors.png', 'Четыре наименьших IoU на тесте')
    return {'mean_image_iou': sum(row['iou'] for row in rows) / len(rows),
            'mean_image_dice': sum(row['dice'] for row in rows) / len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('Число эпох должно быть положительным')
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    datasets, metadata = load_data()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = UNet()
    validation = DataLoader(datasets[1], batch_size=8)
    if not args.evaluate:
        loader = DataLoader(datasets[0], batch_size=8, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        record = {'history': [], 'data': metadata, 'initial_validation': run_epoch(model, validation),
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'platform': platform.platform(), 'device': 'cpu', 'threads': 4}}
        best = -1
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, loader, optimizer)
            val = run_epoch(model, validation)
            record['history'].append({'epoch': epoch, 'train': train, 'validation': val})
            if val['iou'] > best:
                best = val['iou']
                record['best_epoch'] = epoch
                torch.save({'state_dict': model.state_dict(), 'data': metadata}, output / 'model.pt')
            print(f"Эпоха {epoch}/{args.epochs}: loss={train['loss']:.4f}; val_loss={val['loss']:.4f}; val_IoU={val['iou']:.4f}", flush=True)
        record['training_seconds'] = round(time.perf_counter() - start, 2)
        save_json(output / 'history.json', record)
    if not (output / 'model.pt').exists():
        raise SystemExit('Нет модели. Сначала выполните обучение')
    checkpoint = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
    if checkpoint['data'] != metadata:
        raise SystemExit('Корпус или разбиение не совпадают с обучением')
    model.load_state_dict(checkpoint['state_dict'])
    metrics = {'test': run_epoch(model, DataLoader(datasets[2], batch_size=8)),
               'parameters': sum(p.numel() for p in model.parameters()), 'threshold': .5,
               'data': metadata}
    metrics['test'].update(evaluate_examples(model, datasets[2], metadata['split']['test'], output))
    counts = metrics['test']['pixels']
    metrics['all_background'] = scores(0, 0, counts['tp'] + counts['fn'], counts['tn'] + counts['fp'])
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'split.json', metadata)
    history_path = output / 'history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))['history']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, metric, label in zip(axes, ['loss', 'iou'], ['BCE + Dice loss', 'IoU человека']):
            for key, title in [('train', 'Обучение'), ('validation', 'Валидация')]:
                axis.plot([r['epoch'] for r in history], [r[key][metric] for r in history], label=title)
            axis.set(xlabel='Эпоха', ylabel=label)
            axis.legend(); axis.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output / 'learning_curves.png', dpi=160); plt.close(fig)
    print(json.dumps(metrics['test'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

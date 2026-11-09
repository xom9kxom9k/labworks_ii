"""Посимвольная генерация текста с помощью GRU на Tiny Shakespeare."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42


class TextGRU(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 64)
        self.gru = nn.GRU(64, 256, batch_first=True)
        self.output = nn.Linear(256, vocabulary_size)

    def forward(self, tokens, hidden=None):
        features, hidden = self.gru(self.embedding(tokens), hidden)
        return self.output(features), hidden


def make_windows(tokens, length=100):
    count = (len(tokens) - 1) // length
    if count < 1:
        raise ValueError('Недостаточно текста для одного окна')
    x = tokens[:count * length].reshape(count, length)
    y = tokens[1:count * length + 1].reshape(count, length)
    return TensorDataset(x, y)


def load_data():
    path = ROOT / 'data/input.txt'
    if not path.exists():
        raise SystemExit('Нет data/input.txt. Выполните python prepare_data.py')
    text = path.read_text(encoding='utf-8')
    # Словарь определяется по обучающей части, чтобы не читать отложенные данные.
    first, second = int(len(text) * .8), int(len(text) * .9)
    vocabulary = sorted(set(text[:first]))
    mapping = {char: index for index, char in enumerate(vocabulary)}
    chunks = [text[:first], text[first:second], text[second:]]
    if set(text) - set(vocabulary):
        raise ValueError('В отложенных частях есть неизвестные символы')
    datasets = [make_windows(torch.tensor([mapping[c] for c in chunk])) for chunk in chunks]
    metadata = {'characters': len(text), 'vocabulary_size': len(vocabulary),
                'split_characters': list(map(len, chunks)),
                'windows': [len(part) for part in datasets],
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return vocabulary, datasets, metadata


def run_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in loader:
            logits, _ = model(inputs)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item() * targets.numel()
            count += targets.numel()
    return loss_sum / count


@torch.no_grad()
def generate(model, vocabulary, prompt, length=600, temperature=.8, seed=SEED):
    if not prompt or set(prompt) - set(vocabulary):
        raise ValueError('Затравка должна быть непустой и содержать символы словаря')
    if temperature <= 0 or not math.isfinite(temperature) or length < 0:
        raise ValueError('Нужны конечная положительная температура и неотрицательная длина')
    model.eval()
    mapping = {char: index for index, char in enumerate(vocabulary)}
    tokens = torch.tensor([[mapping[char] for char in prompt]])
    hidden = None
    result = prompt
    generator = torch.Generator().manual_seed(seed)
    for _ in range(length):
        logits, hidden = model(tokens, hidden)
        probabilities = (logits[0, -1] / temperature).softmax(0)
        token = torch.multinomial(probabilities, 1, generator=generator)
        result += vocabulary[token.item()]
        tokens = token.reshape(1, 1)
    return result


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--prompt', default='ROMEO:\n')
    parser.add_argument('--length', type=int, default=600)
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.epochs < 1 or args.length < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Проверьте число эпох, длину и температуру')
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    vocabulary, datasets, metadata = load_data()
    if not args.prompt or set(args.prompt) - set(vocabulary):
        parser.error('Затравка пуста или содержит неизвестные символы')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = TextGRU(len(vocabulary))
    validation = DataLoader(datasets[1], batch_size=64)
    if not args.evaluate:
        loader = DataLoader(datasets[0], batch_size=64, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        record = {'history': [], 'seed': SEED, 'data': metadata,
                  'initial_validation_loss': run_epoch(model, validation),
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'platform': platform.platform(), 'device': 'cpu', 'threads': 4}}
        best = float('inf')
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, loader, optimizer)
            validation_loss = run_epoch(model, validation)
            record['history'].append({'epoch': epoch, 'train_loss': train_loss,
                                      'validation_loss': validation_loss})
            if validation_loss < best:
                best = validation_loss
                record['best_epoch'] = epoch
                torch.save({'state_dict': model.state_dict(), 'vocabulary': vocabulary,
                            'data_sha256': metadata['sha256']}, output / 'model.pt')
            print(f'Эпоха {epoch}/{args.epochs}: loss={train_loss:.4f}; val_loss={validation_loss:.4f}', flush=True)
        record['training_seconds'] = round(time.perf_counter() - start, 2)
        save_json(output / 'history.json', record)
    if not (output / 'model.pt').exists():
        raise SystemExit('Нет весов модели. Сначала выполните обучение')
    checkpoint = torch.load(output / 'model.pt', map_location='cpu', weights_only=True)
    if checkpoint['vocabulary'] != vocabulary or checkpoint['data_sha256'] != metadata['sha256']:
        raise SystemExit('Корпус не совпадает с использованным при обучении')
    model.load_state_dict(checkpoint['state_dict'])
    test_loss = run_epoch(model, DataLoader(datasets[2], batch_size=64))
    metrics = {'test_loss': test_loss, 'test_perplexity': math.exp(test_loss),
               'parameters': sum(p.numel() for p in model.parameters()), 'data': metadata}
    save_json(output / 'metrics.json', metrics)
    sample = generate(model, vocabulary, args.prompt, args.length, args.temperature)
    (output / 'generated.txt').write_text(sample, encoding='utf-8')
    samples = [{'temperature': t, 'prompt': args.prompt, 'seed': SEED,
                'text': generate(model, vocabulary, args.prompt, args.length, t)} for t in (.5, .8, 1.1)]
    save_json(output / 'samples.json', samples)
    history_path = output / 'history.json'
    if history_path.exists():
        rows = json.loads(history_path.read_text(encoding='utf-8'))['history']
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows], label='Обучение')
        axis.plot([r['epoch'] for r in rows], [r['validation_loss'] for r in rows], label='Валидация')
        axis.set(xlabel='Эпоха', ylabel='Cross entropy, нат/символ', title='Обучение посимвольной GRU')
        axis.legend()
        axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / 'learning_curves.png', dpi=160)
        plt.close(fig)
    print(f'Тест: loss={test_loss:.4f}; perplexity={math.exp(test_loss):.3f}\n\n{sample}', flush=True)


if __name__ == '__main__':
    main()

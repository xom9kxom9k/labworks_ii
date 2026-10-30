"""Обучение CNN и визуализация классификации Fashion-MNIST."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import FashionMNIST

CLASSES = ["Футболка", "Брюки", "Свитер", "Платье", "Пальто",
           "Сандалии", "Рубашка", "Кроссовки", "Сумка", "Ботинки"]
SEED = 42
RESULTS = ROOT / "results"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.layers(images)


def load_data():
    datasets = []
    for train in (True, False):
        try:
            source = FashionMNIST(ROOT / "data", train=train, download=False)
        except RuntimeError as error:
            raise SystemExit("Нет локального датасета. Сначала выполните python prepare_data.py") from error
        images = source.data.unsqueeze(1).float() / 255.0
        datasets.append(TensorDataset(images, source.targets))
    train, validation = random_split(
        datasets[0], [54000, 6000], generator=torch.Generator().manual_seed(SEED)
    )
    return train, validation, datasets[1]


def run_epoch(model, loader, criterion, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = total = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train_model(model, train, validation, epochs):
    loader = DataLoader(train, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_loss = float("inf")
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loader, criterion, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), RESULTS / "model.pt")
        print(f"Эпоха {epoch}/{epochs}: loss={train_metrics['loss']:.4f}, "
              f"accuracy={train_metrics['accuracy']:.2%}; "
              f"val_loss={validation_metrics['loss']:.4f}, "
              f"val_accuracy={validation_metrics['accuracy']:.2%}", flush=True)
    record = {
        "history": history, "best_epoch": best_epoch, "seed": SEED,
        "training_seconds": round(time.perf_counter() - start, 2),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads()},
    }
    (RESULTS / "history.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ["loss", "accuracy"], ["Функция потерь", "Доля верных ответов"]):
        for split, label in [("train", "Обучение"), ("validation", "Валидация")]:
            axis.plot([row["epoch"] for row in history], [row[split][metric] for row in history], label=label)
        axis.set(xlabel="Эпоха", ylabel=title, title=title)
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_examples(images, labels, predictions, probabilities, indices, filename, title):
    indices = indices[:24]
    if not len(indices):
        return
    rows = (len(indices) + 5) // 6
    fig, axes = plt.subplots(rows, 6, figsize=(15, rows * 2.5), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, indices):
        actual, predicted = labels[index].item(), predictions[index].item()
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Ответ: {CLASSES[actual]}\nПрогноз: {CLASSES[predicted]}\n"
                       f"p = {probabilities[index, predicted]:.1%}",
                       fontsize=9, color="darkgreen" if actual == predicted else "firebrick")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULTS / filename, dpi=150)
    plt.close(fig)


def evaluate(model, test):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for images, _ in DataLoader(test, batch_size=256):
            probabilities.append(model(images).softmax(1))
    probabilities = torch.cat(probabilities)
    predictions = probabilities.argmax(1)
    images, labels = test.tensors
    matrix = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    test_metrics = run_epoch(model, DataLoader(test, batch_size=256), nn.CrossEntropyLoss())
    metrics = {
        "test": test_metrics,
        "samples": {"train": 54000, "validation": 6000, "test": len(test)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classes": CLASSES,
        "per_class_accuracy": (matrix.diag() / matrix.sum(1)).tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(10, 9))
    heatmap = axis.imshow(matrix.numpy(), cmap="Blues")
    fig.colorbar(heatmap, ax=axis)
    axis.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Предсказанный класс", ylabel="Истинный класс", title="Матрица ошибок, тестовая выборка")
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(10):
        for col in range(10):
            value = matrix[row, col].item()
            axis.text(col, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max().item() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    plot_examples(images, labels, predictions, probabilities, range(24), "predictions.png",
                  "Первые 24 изображения тестовой выборки")
    mistakes = torch.where(predictions != labels)[0].tolist()
    plot_examples(images, labels, predictions, probabilities, mistakes, "errors.png",
                  "Первые 24 ошибки на тестовой выборке")
    print(f"Тест: accuracy={test_metrics['accuracy']:.2%}, loss={test_metrics['loss']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8, help="Количество эпох, по умолчанию 8")
    parser.add_argument("--evaluate", action="store_true", help="Проверить готовую модель без обучения")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Количество эпох должно быть положительным")
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    train, validation, test = load_data()
    model = CNN()
    if not args.evaluate:
        train_model(model, train, validation, args.epochs)
    if not (RESULTS / "model.pt").is_file():
        raise SystemExit("Модель не найдена. Сначала запустите python main.py")
    model.load_state_dict(torch.load(RESULTS / "model.pt", map_location="cpu", weights_only=True))
    history_path = RESULTS / "history.json"
    if history_path.exists():
        plot_history(json.loads(history_path.read_text(encoding="utf-8"))["history"])
    evaluate(model, test)
    print(f"Результаты сохранены: {RESULTS}")


if __name__ == "__main__":
    main()

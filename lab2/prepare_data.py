"""Однократная загрузка Fashion-MNIST на диск перед обучением."""

from pathlib import Path

from torchvision.datasets import FashionMNIST


ROOT = Path(__file__).resolve().parent


def main():
    # HTTPS-копия авторов датасета; torchvision проверяет MD5 каждого файла.
    FashionMNIST.mirrors = [
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
    ]
    for train in (True, False):
        dataset = FashionMNIST(ROOT / "data", train=train, download=True)
        print(f"{'Обучающая' if train else 'Тестовая'} выборка: {len(dataset)} изображений")
    print(f"Данные сохранены: {ROOT / 'data'}")


if __name__ == "__main__":
    main()

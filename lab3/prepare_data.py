"""Загрузка открытого корпуса Tiny Shakespeare на диск."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


def main():
    path = Path(__file__).resolve().parent / 'data/input.txt'
    data = urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Контрольная сумма не совпала; файл не сохранён')
    data.decode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f'Сохранено {len(data)} байт: {path}')


if __name__ == '__main__':
    main()

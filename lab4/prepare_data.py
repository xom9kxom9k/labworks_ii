"""Загрузка и распаковка Penn-Fudan Pedestrian с проверкой архива."""
import hashlib
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
SHA256 = '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2'


def main():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    path = folder / 'PennFudanPed.zip'
    if path.exists():
        data = path.read_bytes()
    else:
        data = urlopen(URL, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit('Контрольная сумма архива не совпала')
    if not path.exists():
        path.write_bytes(data)
    count = 0
    with ZipFile(path) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if len(parts) == 3 and parts[0] == 'PennFudanPed' and parts[1] in ('PNGImages', 'PedMasks') and parts[2].endswith('.png'):
                target = folder.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
                count += 1
        for name in archive.namelist():
            if name.lower() == 'pennfudanped/readme.txt':
                (folder / 'PennFudanPed/README.txt').write_bytes(archive.read(name))
    if count != 340:
        raise SystemExit(f'Ожидалось 340 изображений и масок, получено {count}')
    print(f'Распаковано {count} PNG-файлов')


if __name__ == '__main__':
    main()

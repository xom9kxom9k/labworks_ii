# Лабораторная работа № 1. Классификация изображений

Дисциплина: «Системы искусственного интеллекта», 7 семестр.

Решение на PyTorch: свёрточная нейронная сеть распознаёт 10 категорий одежды
из Fashion-MNIST. Датасет лежит в `data/FashionMNIST/raw/`, обучение не обращается
к сети. Аугментация не используется, её можно добавить во второй работе.

## Файлы

| Файл | Назначение |
| --- | --- |
| `prepare_data.py` | Однократная загрузка датасета с проверкой MD5 |
| `main.py` | Модель, обучение, проверка и построение графиков |
| `test_main.py` | Проверка формы выхода, градиентов и разделения выборок |
| `results/model.pt` | Веса лучшей модели по потерям на валидации |
| `results/history.json` | Метрики по эпохам и условия обучения |
| `results/metrics.json` | Итоговые метрики и матрица ошибок |
| `results/learning_curves.png` | Графики обучения |
| `results/predictions.png` | Предсказания на первых 24 тестовых изображениях |
| `results/errors.png` | Первые 24 ошибки классификации |
| `results/confusion_matrix.png` | Матрица ошибок на всём тестовом наборе |
| `Отчет.md` | Описание выполнения и результатов |

## Запуск в Windows

Рекомендуется 64-разрядный Python 3.13. Склонируйте репозиторий или скачайте его целиком через Code → Download ZIP.
Откройте терминал в папке `lab1` и выполните:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --evaluate
```

Последняя команда использует готовую модель и локальный датасет, пересчитывает
метрики и сохраняет изображения в `results`. После установки зависимостей
интернет для проверки и обучения не нужен.

Повторное обучение с нуля (перезаписывает результаты и веса):

```bat
.venv\Scripts\python main.py --epochs 8
```

Если датасет удалён, сначала выполните:

```bat
.venv\Scripts\python prepare_data.py
```

Проверка программы:

```bat
.venv\Scripts\python -m unittest -v test_main.py
```

В macOS/Linux вместо `.venv\Scripts\python` используется `.venv/bin/python`.
Обучение выполняется на CPU в четыре потока. Число эпох по умолчанию: 8,
размер обучающего пакета: 128. Время зависит от компьютера. Seed равен 42;
между версиями библиотек и платформами результаты могут немного отличаться.
Картинки сохраняются в файлы, отдельное окно не открывается.

## Источники

- [Задание лабораторной](https://github.com/ulstu/ml/blob/master/ml_course_ru/assignments/ai_systems2/lab_tasks.md).
- [Fashion-MNIST: данные, описание классов и лицензия MIT](https://github.com/zalandoresearch/fashion-mnist).
- [PyTorch: загрузка данных и Fashion-MNIST](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Датасет: Han Xiao, Kashif Rasul, Roland Vollgraf, «Fashion-MNIST: a Novel
Image Dataset for Benchmarking Machine Learning Algorithms», 2017,
[arXiv:1708.07747](https://arxiv.org/abs/1708.07747).

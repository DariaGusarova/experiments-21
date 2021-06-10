# experiments-21

Здесь собраны основные эксперименты по использованию оптимизации второго порядка, а именно основанной на приближении матрицы информации Фишера с помощью произведения Кронекера, для Трансформера (из оригинальной статьи https://arxiv.org/abs/1706.03762) над WMT14 English-German.

По файлам:
1. Байзлайн - это оптимизация с помощью Адама: baseline.ipynb
2. Предобработка данных с помощью вычищения пунктуации и BPE (Byte-Pair Encoding): data_preprocessing.ipynb
3. Вспомогательный файл с блоками Трансформера: transformer.py
4. Вспомогательный файл с функциями для оптимизации: training.py
5. Предобработанные данные и предобученные модели для Word2Vec тут https://drive.google.com/drive/folders/1DKEXyw7kG04HjC2VLQdjbjpr4xAh5F_A
6. Результаты для Адама тут https://drive.google.com/drive/folders/1dKJAxzkXWa-abCNWhA6rVnwe59I_iaIH?usp=sharing



import csv
import pandas as pd

from os import makedirs
from os.path import join, exists
from sklearn.model_selection import StratifiedKFold


def load_turkish_stop_words():
    with open("ttc3600_stop_words.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def read_csv_dataset(csv_path):
    stop_words = load_turkish_stop_words()

    news = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            category = line[0]
            text = line[1]

            words = [w for w in text.split(" ") if w not in stop_words]
            news.append((category, words))
    return news


def stratified_k_fold(csv_path, target_dir, k=10):
    df = pd.read_csv(csv_path)
    skf = StratifiedKFold(n_splits=k)
    target = df.loc[:, "category"]

    fold_no = 1
    for train_index, test_index in skf.split(df, target):
        train = df.loc[train_index, :]
        test = df.loc[test_index, :]

        tmp = join(target_dir, "fold%02d" % fold_no)
        if not exists(tmp):
            makedirs(tmp)

        train.to_csv(join(tmp, "train.csv"), index=False)
        test.to_csv(join(tmp, "test.csv"), index=False)

        fold_no += 1

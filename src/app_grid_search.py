import argparse
import pandas as pd
import time

from news2vec import News2VecTrainer, News2VecClassifier
from os import listdir
from os.path import join

CLI = argparse.ArgumentParser()
CLI.add_argument("fold_dir", help="Absolute path of the skfold directory")
CLI.add_argument("model_path", help="Absolute path of the classification model")
CLI.add_argument("accuracy_path", help="Absolute path of the accuracy results csv file")
CLI.add_argument("time_path", help="Absolute path of the time results csv file")


def export_time(tuples, path):
    d = []
    for t in tuples:
        d.append({"FOLD": t[0], "DM": t[1], "VS": t[2], "E": t[3], "TIME": t[4]})

    df = pd.DataFrame(d)
    overall = df.groupby(["DM", "VS", "E"])[["TIME"]].agg(["sum"])
    overall.to_csv(path)


def export_accuracy(tuples, path):
    d = []
    for t in tuples:
        d.append({"FOLD": t[0], "DM": t[1], "VS": t[2], "E": t[3], "N": t[4], "ACC": t[5]})

    df = pd.DataFrame(d)
    overall = df.groupby(["DM", "VS", "E", "N"])[["ACC"]].agg(["mean"])
    overall.to_csv(path)


if __name__ == '__main__':
    args = CLI.parse_args()
    fold_dir, model_path, accuracy_path, time_path = args.fold_dir, args.model_path, args.accuracy_path, args.time_path

    dm_options = [0, 1]
    vs_options = range(40, 401, 40)
    epoch_options = [30]
    topn_options = range(1, 21)

    accuracy_logs, time_logs = [], []
    for fold in [f for f in sorted(listdir(fold_dir)) if f.startswith("fold")]:
        train_csv, test_csv = join(fold_dir, fold, "train.csv"), join(fold_dir, fold, "test.csv")

        for dm in dm_options:
            for vs in vs_options:
                for e in epoch_options:
                    start = time.time()
                    trainer = News2VecTrainer(train_csv=train_csv, model_path=model_path, dm=dm, vector_size=vs, epochs=e)
                    trainer.train()
                    elapsed = time.time() - start

                    time_log = (fold, dm, vs, e, elapsed)
                    time_logs.append(time_log)
                    print("%s\tdm=%d\tvs=%d\te=%d\ttime=%f" % time_log)

                    classifier = News2VecClassifier(test_csv=test_csv, model_path=model_path)
                    classifier.load()

                    for n in topn_options:
                        accuracy = classifier.batch_predict(n)
                        accuracy_tuple = (fold, dm, vs, e, n, accuracy)
                        print("%s\tdm=%d\tvs=%d\te=%d\tn=%d\tacc=%f" % accuracy_tuple)
                        accuracy_logs.append(accuracy_tuple)

    export_accuracy(accuracy_logs, accuracy_path)
    export_time(time_logs, time_path)

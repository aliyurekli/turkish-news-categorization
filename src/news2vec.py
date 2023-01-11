import csv

from collections import Counter
from ttc3600 import read_csv_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class News2VecTrainer:
    MODEL_FILE = "news2vec.model"
    IDX2CATEGORY_FILE = "idx2category.csv"

    def __init__(self, train_csv, model_path, dm, vector_size, epochs):
        self.train_csv = train_csv
        self.model_path = model_path
        self.dm = dm
        self.vector_size = vector_size
        self.epochs = epochs

    def create_corpus(self):
        train_news = read_csv_dataset(self.train_csv)
        corpus, idx, idx2category = [], 0, dict()
        for category, words in train_news:
            corpus.append(TaggedDocument(words=words, tags=[idx]))
            idx2category[idx] = category
            idx += 1
        return corpus, idx2category

    def save_model(self, model):
        temp = self.model_path + "/" + News2VecTrainer.MODEL_FILE
        print("Writing model file: %s" % temp)
        model.save(temp)

    def save_idx2category(self, save_idx2category):
        temp = self.model_path + "/" + News2VecTrainer.IDX2CATEGORY_FILE
        print("Writing corpus mapping file: %s" % temp)
        with open(temp, "w") as o:
            writer = csv.writer(o)
            for k, v in save_idx2category.items():
                writer.writerow([k, v])

    def train(self):
        print("Training...")
        corpus, idx2category = self.create_corpus()

        model = Doc2Vec(dm=self.dm, vector_size=self.vector_size, epochs=self.epochs, min_count=2)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

        self.save_model(model)
        self.save_idx2category(idx2category)


class News2VecClassifier:
    def __init__(self, test_csv, model_path):
        self.test_csv = test_csv
        self.model_path = model_path

        # doc2vec model variables
        self.news_model = None
        self.idx2category = dict()

    def load(self):
        print("Loading...")
        self.__reload_news2vec()

    def __reload_news2vec(self):
        self.news_model: Doc2Vec = Doc2Vec.load(self.model_path + "/" + News2VecTrainer.MODEL_FILE)

        with open(self.model_path + "/" + News2VecTrainer.IDX2CATEGORY_FILE) as o:
            reader = csv.reader(o)
            for row in reader:
                self.idx2category[int(row[0])] = row[1]

    def __predict(self, words, n):
        inferred_vector = self.news_model.infer_vector(words)
        similar_news = self.news_model.dv.most_similar([inferred_vector], topn=n)

        hits = Counter()
        for sn in similar_news:
            category = self.idx2category[sn[0]]
            hits[category] += 1

        return hits.most_common(n=1)[0][0]

    def batch_predict(self, n):
        test_news = read_csv_dataset(self.test_csv)

        correct, incorrect = 0, 0
        for category, words in test_news:
            prediction = self.__predict(words, n)

            if prediction == category:
                correct += 1
            else:
                incorrect += 1

            print(category + "\t" + prediction)

        accuracy = correct / (correct + incorrect)
        return accuracy

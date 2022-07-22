import torch
from torch.utils.data import Dataset


CLEAN_TEXT_2020_SUBS = [f'../data/solid/folds/2020_cleanText_fold{i}.txt' for i in range(20)]


class TweetWarehouse:
    def __init__(self):
        self.id2tweet = {}
        self.id2label = {}

    def __len__(self):
        return len(self.id2label)

    def put_in(self, tid, tweet, label):
        self.id2tweet[tid] = tweet
        self.id2label[tid] = label

    def get_tweet(self, tid):
        return self.id2tweet.get(tid, "Tweet NOT FOUND ERROR")

    def get_label(self, tid):
        return self.id2label.get(tid, "Tweet NOT FOUND ERROR")


def read_from_file(src, warehouse=None):
    if not warehouse:
        warehouse = TweetWarehouse()
    file_path = src
    for t in open(file_path, 'r'):
        try:
            tid, lb, text = t.split("\t\t", maxsplit=3)
        except ValueError:
            print(f"deprecate: {t}")
            continue
        if lb == "0":
            warehouse.put_in(tid, text.strip(), 0)
        elif lb == "1":
            warehouse.put_in(tid, text.strip(), 1)
        else:
            warehouse.put_in(tid, text.strip(), float(lb))
    return warehouse


class DataReader:
    def __init__(self, data_src):
        self.database = read_from_file(src=data_src)

    def get_data(self):
        samples = []
        labels = []
        for tid in self.database.id2label.keys():
            x = self.database.get_tweet(tid)
            y = self.database.get_label(tid)
            samples.append(x)
            labels.append(y)
        return samples, labels


class TextDataset(Dataset):
    def __init__(self, samples_iterable, tokenizer):
        super(TextDataset, self).__init__()
        self.samples = []
        # self.target = target_iterable

        for text in samples_iterable:
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            self.samples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.tensor(self.samples[i], dtype=torch.long)


def main():
    from transformers import BertTokenizer
    X, _ = DataReader(data_src='./toy0.txt').get_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=256)
    data = TextDataset(X, tokenizer)
    # train_loader = Data.DataLoader(TextDataset(X_train, y_train), batch_size=32,
    #                                collate_fn=collector, drop_last=False, num_workers=0)
    sample = next(iter(data))
    print(sample)


if __name__ == '__main__':
    main()

import torch
from torch.utils.data import Dataset

class IobDataset(Dataset):
    def __init__(self, path, tokenizer, label_vocab):
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab
        self._read(path)

    def _read(self, path):
        with open(path, 'r') as f:
            lines = f.read()

        lines = lines.split("\n\n")
        lines = [[token.split('\t') for token in l.split("\n") if token != ""] for l in lines if l != ""]
        #print(lines)
        self.tokens = [[t[0] for t in tokens] for tokens in lines]
        #print(len(self.tokens[0]))
        self.labels = [[l[1] for l in labels] for labels in lines]
        #print(len(self.labels[0]))
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        #print(self.tokens[item])
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokens[item])
        labels = [self.label_vocab[l] for l in self.labels[item]]

        return tokens, labels


def create_label_vocab_from_file(fn):
    vocab = {}
    with open(fn, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line not in vocab:
                vocab[line] = len(vocab)
    return vocab


def my_collate_fn(batch):
    tokens, labels = list(zip(*batch))

    return tokens, labels


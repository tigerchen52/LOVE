import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


class WordVocabulary(object):
    def __init__(self, train_path, dev_path, test_path, number_normalized):
        self.number_normalized = number_normalized
        self._id_to_word = []
        self._word_to_id = {}
        self._pad = -1
        self._unk = -1
        self.index = 0

        self._id_to_word.append('<PAD>')
        self._word_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1
        self._id_to_word.append('<UNK>')
        self._word_to_id['<UNK>'] = self.index
        self._unk = self.index
        self.index += 1

        with open(train_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

        with open(dev_path, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

        with open(test_path, 'r', encoding='utf-8') as f3:
            lines = f3.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk()

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def items(self):
        return self._word_to_id.items()


class LabelVocabulary(object):
    def __init__(self, filename):
        self._id_to_label = []
        self._label_to_id = {}
        self._pad = -1
        self.index = 0

        self._id_to_label.append('<PAD>')
        self._label_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1

        with open(filename, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]

                    if label not in self._label_to_id:
                        self._id_to_label.append(label)
                        self._label_to_id[label] = self.index
                        self.index += 1

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_label)

    def label_to_id(self, label):
        return self._label_to_id[label]

    def id_to_label(self, cur_id):
        return self._id_to_label[cur_id]


class Alphabet(object):
    def __init__(self, train_path, dev_path, test_path,):
        self._id_to_char = []
        self._char_to_id = {}
        self._pad = -1
        self._unk = -1
        self.index = 0

        self._id_to_char.append('<PAD>')
        self._char_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1

        self._id_to_char.append('<UNK>')
        self._char_to_id['<UNK>'] = self.index
        self._unk = self.index
        self.index += 1

        with open(train_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]

                    chars = list(word)
                    for char in chars:
                        if char not in self._char_to_id:
                            self._id_to_char.append(char)
                            self._char_to_id[char] = self.index
                            self.index += 1

        with open(dev_path, 'r', encoding='utf-8') as f2:
            lines = f2.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]

                    chars = list(word)
                    for char in chars:
                        if char not in self._char_to_id:
                            self._id_to_char.append(char)
                            self._char_to_id[char] = self.index
                            self.index += 1

        with open(test_path, 'r', encoding='utf-8') as f3:
            lines = f3.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]

                    chars = list(word)
                    for char in chars:
                        if char not in self._char_to_id:
                            self._id_to_char.append(char)
                            self._char_to_id[char] = self.index
                            self.index += 1

    def pad(self):
        return self._pad

    def unk(self):
        return self._unk

    def size(self):
        return len(self._id_to_char)

    def char_to_id(self, char):
        if char in self._char_to_id:
            return self._char_to_id[char]
        return self.unk()

    def id_to_char(self, cur_id):
        return self._id_to_char[cur_id]

    def items(self):
        return self._char_to_id.items()


def my_collate(key, batch_tensor):
    if key == 'char':
        batch_tensor = pad_char(batch_tensor)
        return batch_tensor
    else:
        word_seq_lengths = torch.LongTensor(list(map(len, batch_tensor)))
        _, word_perm_idx = word_seq_lengths.sort(0, descending=True)
        batch_tensor.sort(key=lambda x: len(x), reverse=True)
        tensor_length = [len(sq) for sq in batch_tensor]
        batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=0)
        return batch_tensor, tensor_length, word_perm_idx


def my_collate_fn(batch):
    return {key: my_collate(key, [d[key] for d in batch]) for key in batch[0]}


def pad_char(chars):
    batch_size = len(chars)
    max_seq_len = max(map(len, chars))
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len)).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    return char_seq_tensor


def load_pretrain_emb(embedding_path, embedd_dim=100):

    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if not embedd_dim + 1 == len(tokens):
                continue
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim


def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim=100):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, embedd_dim)
    vocab_size = word_vocab.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_vocab.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / vocab_size))
    return pretrain_emb


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask


if __name__ == '__main__':
    train_path = 'input/train.txt'
    dev_path = 'input/valid.txt'
    test_path = 'input/test.txt'
    word_vocab = WordVocabulary(train_path, dev_path, test_path, True)
    print(word_vocab)
    # pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, word_embed_dim)
    pretrain_word_embedding = build_pretrain_embedding('input/glove.6B.100d.txt', word_vocab, 100)
    print(pretrain_word_embedding)
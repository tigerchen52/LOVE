from registry import register
from functools import partial
import torch
import random
from torch.utils.data import DataLoader
from utils import load_dataset, TextData, repre_word, load_neg_samples
from attacks import get_random_attack
registry = {}
register = partial(register, registry=registry)


@register('simple')
class SimpleLoader():
    def __init__(self, args, TOKENIZER):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dim = args.emb_dim
        self.input_type = args.input_type
        self.lowercase = args.lowercase
        self.tokenizer = TOKENIZER

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))
        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words)):
            aug_word = batch_words[index]
            repre, repre_ids = repre_word(aug_word, self.tokenizer, rtype=self.input_type)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_oririn_repre, batch_aug_repre_ids, mask

    def __call__(self, data_path, neg_sample_path=''):
        dataset, _ = load_dataset(path=data_path, DIM=self.dim, lower=self.lowercase)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


@register('aug')
class SimpleLoader():
    def __init__(self, args, TOKENIZER):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.dim = args.emb_dim
        self.input_type = args.input_type
        self.lowercase = args.lowercase
        self.tokenizer = TOKENIZER

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))

        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words)):
            aug_word = get_random_attack(batch_words[index])
            repre, repre_ids = repre_word(aug_word,  self.tokenizer, rtype=self.input_type)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_oririn_repre, batch_aug_repre_ids, mask

    def __call__(self, data_path):
        dataset, _ = load_dataset(path=data_path, DIM=self.dim, lower=self.lowercase)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


@register('hard')
class SimpleLoader():
    def __init__(self, args, TOKENIZER):
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.hard_neg_numbers = args.hard_neg_numbers
        self.dim = args.emb_dim
        self.neg_samples = load_neg_samples(args.hard_neg_path)
        self.input_type = args.input_type
        self.lowercase = args.lowercase
        self.all_words = list(self.neg_samples.keys())
        self.tokenizer = TOKENIZER

        # to load in the call function
        self.emb = None

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))

        batch_words_with_hards, batch_repre_with_hards = list(), list()
        for word in batch_words:
            if word in self.neg_samples:
                neg_words = self.neg_samples[word]
            else:
                neg_words = []
            if len(neg_words) >= self.hard_neg_numbers:
                batch_hards = list(random.sample(neg_words, self.hard_neg_numbers))
            else:
                sum_words = list(random.sample(self.all_words, self.hard_neg_numbers - len(neg_words)))
                batch_hards = list(neg_words + sum_words)
            batch_words_with_hards.append(word)
            batch_words_with_hards.extend(batch_hards)
            batch_repre_with_hards.append(self.emb[word])
            for w in batch_hards:
                if w not in self.emb:
                    print('this word {a} does not in vocab'.format(a=w))
            batch_repre_with_hards.extend([self.emb[w] if w in self.emb else self.emb['<unk>'] for w in batch_hards])

        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words_with_hards)):
            aug_word = get_random_attack(batch_words_with_hards[index])
            repre, repre_ids = repre_word(aug_word, self.tokenizer, id_mapping=None, rtype=self.input_type)
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = batch_words_with_hards + aug_words
        batch_repre_with_hards = torch.FloatTensor(batch_repre_with_hards)
        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_repre_with_hards, batch_aug_repre_ids, mask

    def __call__(self, data_path):
        dataset, self.emb = load_dataset(path=data_path, DIM=self.dim, lower=self.lowercase)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // (2 * (self.hard_neg_numbers+1)), shuffle=self.shuffle, collate_fn=self.collate_fn)
        return train_iterator
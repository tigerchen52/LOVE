# reference repo
# https://github.com/cswangjiawei/pytorch-NER
# https://github.com/jiesutd/NCRFpp

import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, Alphabet, build_pretrain_embedding, my_collate_fn, lr_decay
import time
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_model, evaluate

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
parser.add_argument('--word_embed_dim', type=int, default=300)
parser.add_argument('--word_hidden_dim', type=int, default=100)
parser.add_argument('--char_embedding_dim', type=int, default=30)
parser.add_argument('--char_hidden_dim', type=int, default=50)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pretrain_embed_path', default='glove.6B.100d.txt')
parser.add_argument('--savedir', default='output/')
parser.add_argument('--batch_size', type=int, default=768)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--feature_extractor', choices=['lstm', 'cnn', 'linear'], default='lstm')
parser.add_argument('--use_char', type=bool, default=False)
parser.add_argument('--train_path', default='evaluation/train.txt')
parser.add_argument('--dev_path', default='evaluation/valid.txt')
parser.add_argument('--test_path', default='evaluation/test.txt')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--number_normalized', type=bool, default=True)
parser.add_argument('--use_crf', type=bool, default=True)


def single_run(args, word_vocab, pretrain_word_embedding):

    use_gpu = torch.cuda.is_available()
    print('use_crf:', args.use_crf)
    print('emb dim:', args.word_embed_dim)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    eval_path = "evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'

    model_name = args.savedir + '/' + args.feature_extractor + str(args.use_char) + str(args.use_crf)
    #word_vocab = WordVocabulary(args.train_path, args.dev_path, args.test_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)
    alphabet = Alphabet(args.train_path, args.dev_path, args.test_path)

    # emb_begin = time.time()
    # pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    # emb_end = time.time()
    # emb_min = (emb_end - emb_begin) % 3600 // 60
    # print('build pretrain embed cost {}m'.format(emb_min))

    train_dataset = MyDataset(args.train_path, word_vocab, label_vocab, alphabet, args.number_normalized)
    dev_dataset = MyDataset(args.dev_path, word_vocab, label_vocab, alphabet, args.number_normalized)
    test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, alphabet, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim, alphabet.size(),
                             args.char_embedding_dim, args.char_hidden_dim,
                             args.feature_extractor, label_vocab.size(), args.dropout,
                             pretrain_embed=pretrain_word_embedding, use_char=args.use_char, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_begin = time.time()
    print('train begin', '-' * 50)
    print()
    print()

    writer = SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate(dev_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set:', best_f1)
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-' * 50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    print('-' * 50)

    model.load_state_dict(torch.load(model_name))
    test_acc = evaluate(test_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
    print('test acc on test set:', test_acc)
    return test_acc


if __name__ == '__main__':
    args = parser.parse_args()
    word_vocab = WordVocabulary(args.train_path, args.dev_path, args.test_path, args.number_normalized)
    with open('output/words.txt', 'w', encoding='utf8')as f:
        f.write('\n'.join([str.lower(w) for w in word_vocab._id_to_word]))

    args.pretrain_embed_path = 'output/love.emb'
    args.word_embed_dim = 300

    pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    single_run(args, word_vocab, pretrain_word_embedding)



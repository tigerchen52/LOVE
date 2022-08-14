import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import argparse


parser = argparse.ArgumentParser(description='CNN Text Classification')
parser.add_argument('--word_embed_dim', type=int, default=300)
parser.add_argument('--cnn_filter_num', type=int, default=100)
parser.add_argument('--cnn_kernel_size', type=list, default=[3,4,5])
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--pretrain_embed_path', default='output/love.emb')
parser.add_argument('--model_path', default='output/model.pt')
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--vocab_path', default='output/words.txt')
parser.add_argument('--train_path', default='evaluation/train.tsv')
parser.add_argument('--dev_path', default='evaluation/dev.tsv')
parser.add_argument('--test_path', default='evaluation/test.tsv')
parser.add_argument('--seed', type=int, default=42)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes,
                 dropout, pre_embedding=None, PAD_IDX=0):
        super().__init__()

        # word embedding
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=PAD_IDX)
        self.embedding.weight.data.copy_(torch.from_numpy(pre_embedding))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, word):

        embedded = self.embedding(word)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for words, labels in iterator:
        optimizer.zero_grad()

        predictions = model(words).squeeze(1)

        loss = criterion(predictions, labels)

        acc = binary_accuracy(predictions, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for words, labels in iterator:
            predictions = model(words).squeeze(1)

            loss = criterion(predictions, labels)

            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def gen_vocabulary(doc_files, vocab_file, max_size=30000):
    '''
    generate a word list given an input corpus
    :param data_file:
    :param vocab_file:
    :return:
    '''
    word_freq = dict()
    all_sentences = list()
    for file in doc_files:
        with open(file, encoding='utf8')as f:
            sentences = [s.split('\t')[0] for s in f.readlines()]
            all_sentences.extend(sentences)
    print('data_size = {a}'.format(a=len(all_sentences)))
    for sentence in all_sentences:
        tokens = sentence.strip().split(' ')
        for token in tokens:
            if token not in word_freq:
                word_freq[token] = 0
            word_freq[token] += 1

    sorted_word_freq = sorted(word_freq.items(), key=lambda e:e[1], reverse=True)[:max_size]
    sorted_word_freq = [k for k, v in sorted_word_freq]

    with open(vocab_file, 'w', encoding='utf8')as f:
        f.write('\n'.join(sorted_word_freq))

    print('done! The size of vocabulary is {a}.'.format(a=len(sorted_word_freq)))


def load_vocabulary(path, PAD_IDX=0):
    vocab_dict = dict()
    word_list = []
    for index, line in enumerate(open(path, encoding='utf8')):
        row = line.strip()
        w_id = index + 1
        text = row
        vocab_dict[text] = w_id
        word_list.append(text)
    vocab_dict['<pad>'] = 0
    vocab_dict['<sep>'] = len(vocab_dict)
    vocab_dict['<unk>'] = len(vocab_dict)
    word_list.append('<sep>')
    word_list.append('<unk>')
    word_list.insert(PAD_IDX, '<pad>')
    return vocab_dict, word_list


def pad_sentence(sentence, max_len, PAD_IDX=0):
    if len(sentence) > max_len:
        return sentence[:max_len]
    else:
        for i in range(max_len - len(sentence)):
            sentence.append(PAD_IDX)
        return sentence


def load_sent_by_id(data_file, vocab_dict):
    all_sents = list()
    lables = list()
    with open(data_file, encoding='utf8')as f:
        for index, line in enumerate(f.readlines()):
            row = line.strip().split('\t')
            sentence, label = row[0], row[1]
            words = sentence.strip().split(' ')
            word_ids = [vocab_dict.get(w, len(vocab_dict)-1) for w in words]
            all_sents.append(word_ids)
            lables.append(int(label))
    return {'sentence': all_sents, 'label':lables}


def load_all_embeddings(path, emb_size):
    word_embed_dict = dict()
    with open(path)as f:
        line = f.readline()
        while line:
            row = line.strip().split(' ')
            word_embed_dict[row[0]] = row[1:emb_size+1]
            line = f.readline()
    print('load word embedding, size = {a}'.format(a=len(word_embed_dict)))
    return word_embed_dict


def get_emb_weights(embeddings, word_list, embed_size):
    #torch_weights = torch.zeros(len(word_list), embed_size)
    torch_weights = []
    miss_set = set()
    for index, word in enumerate(word_list):
        if word not in embeddings:
            miss_set.add(word)
            weight = np.zeros(embed_size)
        else:
            weight = [float(v) for v in embeddings[word]]
        #weight[index, :] = torch.from_numpy(np.array(weight))
        torch_weights.append(weight)

    print('in total, {a} words do not have embeddings'.format(a=len(miss_set)))
    print(miss_set)
    return np.array(torch_weights)


def random_emb_value():
    return random.uniform(0, 0)


def process_embedding(embed_path, vocab_list, emb_size, out_path):
    result = []
    w_l = ''
    print('word list size = {a}'.format(a=len(vocab_list)))
    word_embed_dict = load_all_embeddings(embed_path, emb_size)
    print('loaded pre-trained embedding size = {a} '.format(a=len(word_embed_dict)))
    missing_set = set()
    cnt = 0
    for index, word in enumerate(vocab_list):
        if index % 10 == 0:print('process {a} words ...'.format(a=index))
        if word in word_embed_dict:
            emd = word_embed_dict[word]
        else:
            emd = [str(round(random_emb_value(), 5)) for _ in range(emb_size)]
            cnt += 1
            missing_set.add(word)
        result.append(emd)
        w_l += word + ' ' + ' '.join(emd) + '\n'
    print('word size = {a}, {b} words have not embedding value'.format(a=len(vocab_list), b=cnt))
    np.save(out_path, np.array(result))
    print(missing_set)

    with open(out_path, 'w', encoding='utf8')as f:
        f.write(w_l)


class TextData(Dataset):
    def __init__(self, data):
        self.sentence = data['sentence']
        self.label = data['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.sentence[idx], self.label[idx]


def collate_fn(batch_data, PAD_IDX=0):
    sentence, label = list(zip(*batch_data))
    max_len = max([len(sent) for sent in sentence])
    sentence = [sent+[PAD_IDX]*(max_len-len(sent)) for sent in sentence]

    sentence = torch.LongTensor(sentence)
    label = torch.FloatTensor(label)
    return sentence, label


def load_pretrained_emb(emb_path, dim=300, words=None):
    vectors = dict()
    cnt = 0
    for line in open(emb_path, encoding='utf8'):
        cnt += 1
        #if cnt >=1000:return vectors
        row = line.strip().split(' ')
        if len(row) != dim+1:continue
        word = row[0]
        if words is not None and word not in words:continue
        vec = np.array(row[1:], dtype=str)
        if word not in vectors:
            vectors[word] = vec
    return vectors


def fixed_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_sst2(
        args,
):
    word_dict, word_list = load_vocabulary(args.vocab_path)
    word_num = len(word_list)  # len(TEXT.vocab)
    pre_embeddings = load_all_embeddings(args.pretrain_embed_path, args.word_embed_dim)
    print('embedding word size = {a}'.format(a=len(pre_embeddings)))
    pre_embeddings = get_emb_weights(pre_embeddings, word_list, args.word_embed_dim)

    train_data = load_sent_by_id(args.train_path, word_dict)
    dev_data = load_sent_by_id(args.dev_path, word_dict)
    test_data = load_sent_by_id(args.test_path, word_dict)

    train_data = TextData(train_data)
    dev_data = TextData(dev_data)
    test_data = TextData(test_data)
    train_iterator = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iterator = DataLoader(dataset=dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iterator = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CNN(word_num, args.word_embed_dim, args.cnn_filter_num, args.cnn_kernel_size, args.dropout, pre_embedding=pre_embeddings)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_valid_acc = 0.0
    print('epoch = {a}'.format(a=args.epochs))
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, dev_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), args.model_path)

        print(f'Epoch: {e + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load(args.model_path))
    acc = predict(args, model, test_iterator, criterion)
    return acc


def predict(args, model, test_iterator, criterion):
    model.load_state_dict(torch.load(args.model_path))
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print('\t-------------------------------------------------------------')
    print(f'\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print('\t-------------------------------------------------------------')
    return test_acc


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parser.parse_args()
    fixed_seed(seed=args.seed)

    args.pretrain_embed_path = 'output/love.emb'
    args.word_embed_dim = 300
    run_sst2(args)

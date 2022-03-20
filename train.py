import torch
import torch.optim as optim
import sys
import os
import argparse
import tokenization
from torch.optim import lr_scheduler
from loss import registry as loss_f
from loader import registry as loader
from model import registry as Producer
from evaluate import overall


#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-dataset', help='the file of target vectors', type=str, default='data/wiki_100.vec')
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=20)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('-lowercase', help='if only use lower case', type=bool, default=True)
parser.add_argument('-model_type', help='sum, rnn, cnn, attention, pam', type=str, default='pam')
parser.add_argument('-encoder_layer', help='the number of layer of the encoder', type=int, default=1)
parser.add_argument('-merge', help='merge pam and attention layer', type=bool, default=True)
parser.add_argument('-att_head_num', help='the number of attentional head for the pam encoder', type=int, default=1)
parser.add_argument('-loader_type', help='simple, aug, hard', type=str, default='hard')
parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')
parser.add_argument('-input_type', help='mixed, char, sub', type=str, default='mixed')
parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=2e-3)
parser.add_argument('-drop_rate', help='the rate for dropout', type=float, default=0.1)
parser.add_argument('-gamma', help='decay rate', type=float, default=0.97)
parser.add_argument('-emb_dim', help='the dimension of target embeddings (FastText:300; BERT:768)', type=int, default=300)
parser.add_argument('-vocab_path', help='the vocabulary used for training and inference', type=str, default='data/vocab.txt')
parser.add_argument('-hard_neg_numbers', help='the number of hard negatives in each mini-batch', type=int, default=3)
parser.add_argument('-hard_neg_path', help='the file path of hard negative samples ', type=str, default='data/hard_neg_samples.txt')
parser.add_argument('-vocab_size', help='the size of the vocabulart', type=int, default=0)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


def main():
    TOKENIZER = tokenization.FullTokenizer(vocab_file=args.vocab_path, do_lower_case=args.lowercase)
    vocab_size = len(TOKENIZER.vocab)
    args.vocab_size = vocab_size

    data_loader = loader[args.loader_type](args, TOKENIZER)
    train_iterator = data_loader(data_path=args.dataset)

    model = Producer[args.model_type](args)
    print(model)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    criterion = loss_f[args.loss_type]()

    max_acc = 0
    for e in range(args.epochs):
        epoch_loss = 0
        batch_num = 0

        for words, oririn_repre, aug_repre_ids, mask in train_iterator:
            model.train()
            optimizer.zero_grad()
            batch_num += 1

            if batch_num % 1000 == 0:
                print('sample = {b}, loss = {a}'.format(a=epoch_loss/batch_num, b=batch_num*args.batch_size))

            # get produced vectors
            oririn_repre = oririn_repre.cuda()
            aug_repre_ids = aug_repre_ids.cuda()
            mask = mask.cuda()
            aug_embeddings = model(aug_repre_ids, mask)

            # calculate loss
            loss = criterion(oririn_repre, aug_embeddings)
            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print('[ lr rate] = {a}'.format(a=optimizer.state_dict()['param_groups'][0]['lr']))

        print('----------------------')
        print('this is the {a} epoch, loss = {b}'.format(a=e + 1, b=epoch_loss / len(train_iterator)))

        if (e) % 1 == 0:
            model_path = './output/model_{a}.pt'.format(a=e+1)
            torch.save(model.state_dict(), model_path)
            overall(args, model_path=model_path, tokenizer=TOKENIZER)
    return max_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
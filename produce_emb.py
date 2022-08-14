import torch
import tokenization
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import TextData, collate_fn_predict

from train import args
TOKENIZER = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=args.lowercase)
vocab_size = len(TOKENIZER.vocab)
args.vocab_size = vocab_size
model_path = 'output/model_ merge.pt'


def produce(word, batch_size=1):
    dataset = {'origin_word': [word], 'origin_repre':[None]}
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn_predict(x, TOKENIZER, args.input_type))
    model = Producer[args.model_type](args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, mask in train_iterator:
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        emb = model(batch_repre_ids, mask)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))
    return embeddings


def gen_embeddings_for_vocab(vocab_path, emb_path, batch_size=32):
    vocab = [line.strip() for line in open(vocab_path, encoding='utf8')]
    dataset = {'origin_word': vocab, 'origin_repre': [None for _ in range(len(vocab))]}
    dataset = TextData(dataset)
    train_iterator = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda x: collate_fn_predict(x, TOKENIZER, args.input_type))
    model = Producer[args.model_type](args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()

    embeddings = dict()
    for words, _, batch_repre_ids, mask in train_iterator:
        batch_repre_ids = batch_repre_ids.cuda()
        mask = mask.cuda()
        emb = model(batch_repre_ids, mask)
        emb = emb.cpu().detach().numpy()
        embeddings.update(dict(zip(words, emb)))

    wl = open(emb_path, 'w', encoding='utf8')
    for word, embedding in embeddings.items():
        emb_str = ' '.join([str(e) for e in list(embedding)])
        wl.write(word + ' ' + emb_str + '\n')



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #gen_embeddings_for_vocab(vocab_path='extrinsic/rnn_ner/output/words.txt', emb_path='extrinsic/rnn_ner/output/love.emb')
    # gen_embeddings_for_vocab(vocab_path='extrinsic/cnn_text_classification/output/words.txt',
    #                          emb_path='extrinsic/cnn_text_classification/output/love.emb')
    emb = produce('mispelling')
    print(emb)


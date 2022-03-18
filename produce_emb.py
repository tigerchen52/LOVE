import torch
import tokenization
from model import registry as Producer
from torch.utils.data import DataLoader
from utils import TextData, collate_fn_predict

from train import args
TOKENIZER = tokenization.FullTokenizer(vocab_file='data/vocab.txt', do_lower_case=args.lowercase)
vocab_size = len(TOKENIZER.vocab)
args.vocab_size = vocab_size
model_path = 'output/model_in_paper.pt'


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


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    emb = produce('mispelling')
    print(emb)
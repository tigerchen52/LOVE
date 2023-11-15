from transformers import BertTokenizer
import numpy as np
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)




def load_all_embeddings(path, emb_size=300):
    word_embed_dict = dict()
    with open(path)as f:
        line = f.readline()
        while line:
            row = line.strip().split(' ')
            word_embed_dict[row[0]] = [float(v) for v in row[1:emb_size+1]]
            line = f.readline()
    print('load word embedding, size = {a}'.format(a=len(word_embed_dict)))
    return word_embed_dict




def get_token(text):
    sub_tokens = tokenizer.tokenize(text)
    sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)

    length = len(sub_tokens)
    cursor = 0

    indexes = list()
    word_tokens, temp_concat, record_index = list(), '', []
    while cursor < length:
        temp_concat += sub_tokens[cursor]
        record_index.append(sub_ids[cursor])
        if cursor == length - 1 or '##' not in sub_tokens[cursor + 1]:
            word_tokens.append(temp_concat.replace('##', ''))
            indexes.append(record_index)
            temp_concat, record_index = '', []
        cursor += 1

    return word_tokens, indexes




def get_emb_for_text(text, bert_emb=None, embeddings=None,  max_len=50):
    word_tokens, indexes = get_token(text)

    pre_emb = list()
    pad, cls, sep, unk = bert_emb[0], bert_emb[101], bert_emb[102], bert_emb[100]
    pre_emb.append(cls)
    for index, word in enumerate(word_tokens):
        sub_len = len(indexes[index])
        if sub_len == 1:
            pre_emb.append(bert_emb[indexes[index][0]])
        else:

            if word not in embeddings:
                #print('[attention] this {a} does not have an embedding'.format(a=word))
                emb = unk
            else:
                emb = embeddings[word]
            pre_emb.append(emb)

    pre_emb.append(sep)

    if len(pre_emb) >= max_len:
        pre_emb = pre_emb[:max_len-1]

    for i in range(max_len - len(pre_emb)):
        pre_emb.append(pad)
    pre_emb = np.array(pre_emb)
    return pre_emb


if __name__ == '__main__':

    sub_tokens, sub_ids = get_token('a bloated gasbag thesis grotesquely impressed by its own gargantuan aura of self importance')
    print(sub_tokens, sub_ids)




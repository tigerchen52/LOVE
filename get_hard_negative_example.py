import Levenshtein
from multiprocessing import Process
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModel

# if the edit similarity is above the value, the word is regarded to have a similar surface form
ALPHA = 0.70
# how many similar words we collect 
MAXIMUM_NUM = 50
# how many words in each process
PARTITION = 10000
# if edit similarity - semantic similarity > beta, this is a hard negative word, since it have a similar surface form but different meaning
BETA = 0.3

def load_name(file_path):
    '''
    load all names
    :param file_path:
    :return:
    '''
    all_phrase = list()
    return all_phrase


def edit_sim(string_a, string_b):
    return Levenshtein.ratio(string_a, string_b)


def get_phrase_with_high_edit_sim(out_path, e_names, can_names):
    w_f = open(out_path, 'w', encoding='utf8')

    for i in range(len(can_names)):
        if i % 10000 == 0:print('process {a},  {b} lines'.format(a=os.getpid(), b=i))
        input_name = can_names[i]
        temp_can = list()
        for j in range(len(e_names)):
            e_name = e_names[j]
            sim = edit_sim(input_name, e_name)
            if sim > ALPHA:
                temp_can.append(e_name+'__'+str(round(sim, 4)))
            if len(temp_can) > MAXIMUM_NUM or j == (len(e_names)-1):
                w_f.write(input_name+'\t'+'\t'.join(temp_can)+'\n')
                w_f.flush()
                break


def multi_process_edit_sim(file_path):
    e_names = load_name(file_path)
    base_file = 'tmp/edit_hard_negative_{a}.txt'
    file_list = list()

    pro_list = []
    for i in range(0, len(e_names), PARTITION):
        can_names = e_names[i:i+PARTITION]
        out_path = base_file.format(a=i)
        file_list.append(out_path)
        p = Process(target=get_phrase_with_high_edit_sim, args=(out_path, e_names, can_names))
        pro_list.append(p)
        p.start()

    for p in pro_list:
        p.join()

    wf = open('tmp/edit_sim_hard_negative.txt', 'w', encoding='utf8')
    for file in file_list:
        for line in open(file, encoding='utf8'):
            wf.write(line)
            wf.flush()

    print('finished')


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def refine_edit_sim_hard_negative():
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
    model = AutoModel.from_pretrained('intfloat/e5-small-v2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    compute_sim = nn.CosineSimilarity(dim=1)

    wf = open('refined_hard_negative.txt', 'w', encoding='utf8')

    line_cnt = 0
    for line in open('tmp/edit_sim_hard_negative.txt', encoding='utf8'):
        line_cnt += 1
        if line_cnt % 10000 == 0: print('processing {a} lines'.format(a=line_cnt))
        row = line.strip().split('\t')
        phrase = row[0]
        candidates = [e.split('__') for e in row[1:]]

        all_names = [phrase] + [e[0] for e in candidates]
        batch_dict = tokenizer(all_names, max_length=128, padding=True, truncation=True, return_tensors='pt')
        batch_dict = batch_dict.to(device)
        outputs = model(**batch_dict)
        all_vec = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        phrase_vec, candidate_vec = all_vec[0:1], all_vec[1:]
        all_cos_sim = compute_sim(phrase_vec, candidate_vec).cpu().detach().numpy()
        all_cos_sim = (all_cos_sim - all_cos_sim.min()) / (all_cos_sim.max() - all_cos_sim.min())

        temp = list()
        for index, candidate in enumerate(candidates):

            can_name, edit_sim = candidate[0], float(candidate[1])
            cos_sim = all_cos_sim[index]

            if edit_sim - cos_sim > BETA:
                temp.append(can_name)

        wf.write(phrase + '\t' + '\t'.join(temp) + '\n')




if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    multi_process_edit_sim(file_path='')
    refine_edit_sim_hard_negative()


import random
import os
import torch
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from get_emb import load_all_embeddings, get_emb_for_text

parser = argparse.ArgumentParser(description='BERT Text Classification')
parser.add_argument('--word_embed_dim', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--love_embed_path', default='../data/love.emb')
parser.add_argument('--model_path', default='model.pt')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--max_len', type=int, default=50)
parser.add_argument('--vocab_path', default='../data/vocab.txt')
parser.add_argument('--train_path', default='../data/train.tsv')
parser.add_argument('--dev_path', default='../data/dev.tsv')
parser.add_argument('--test_path', default='../data/typo_test_{a}.txt')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--use_love', type=bool, default=True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_data(path):
    sentences, labels = list(), list()
    with open(path, encoding='utf8')as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split('\t')
            sentence, label = row[0], int(row[1])
            sentences.append(sentence)
            labels.append(label)
    return sentences, labels


def flat_accuracy(preds, labels):
    """A function for calculating accuracy scores"""

    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


# Function to get token ids for a list of texts
def encode_fn(text_list, embeddings, bert_emb):

    all_input_ids = []
    all_embs = []
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,
                        add_special_tokens=True,  
                        max_length=max_len,           
                        pad_to_max_length=True,   
                        return_tensors='pt'       
                   )
        all_input_ids.append(input_ids)
        pre_emb = get_emb_for_text(text, bert_emb, embeddings, max_len)
        pre_emb = torch.FloatTensor(pre_emb)
        all_embs.append(pre_emb)

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_embs = torch.stack(all_embs)
    return all_input_ids, all_embs


class TextClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        dim = 768
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.w1 = torch.nn.Linear(dim * 2, dim)
        self.linear = torch.nn.Linear(dim, 768)
        self.sigmoid = torch.nn.Sigmoid()
        self.classifier = torch.nn.Linear(768, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.activiate = torch.nn.Tanh()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            extra_emb=None
    ):
        if use_love:emb = extra_emb
        else:emb=inputs_embeds

        outputs  = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=emb,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

        )

        output = outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1)
        output = output.masked_fill_(~attention_mask, 0).sum(dim=1)
        output = self.dropout(output)
        output = self.activiate(self.linear(output))
        output = self.dropout(output)
        logits = self.classifier(output)
        loss = self.loss_fct(logits, labels)
        return loss, logits



def train(lr=2e-5):
    # Load the pretrained Tokenizer
    print(doc_files)
    train_data, train_label = load_data(doc_files[0])
    dev_data, dev_label = load_data(doc_files[1])
    
    print(len(train_data))

    print('Original Text : ', train_data[2])
    print('Tokenized Text: ', tokenizer.tokenize(train_data[2]))
    print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_data[2])))

    embeddings = load_all_embeddings(path='../data/love.emb', emb_size=emb_dim)

    bert_input = bert_emb.weight.data.to('cpu').numpy()
    train_data, train_emb = encode_fn(train_data, embeddings, bert_input)
    train_label = torch.tensor(train_label)
    dev_data, dev_emb = encode_fn(dev_data, embeddings, bert_input)
    dev_label = torch.tensor(dev_label)


    train_dataloader = DataLoader(TensorDataset(train_data, train_emb, train_label), batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(TensorDataset(dev_data, dev_emb, dev_label), batch_size=batch_size, shuffle=False)
    

    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    max_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            initial_emb = bert_emb(batch[0].to(device))
            #our_emb = batch[1].to(device)
            #initial_emb = initial_emb + our_emb
            loss, logits = model(inputs_embeds=initial_emb, token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 extra_emb=batch[1].to(device),
                                 labels=batch[2].to(device))
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        for i, batch in enumerate(dev_dataloader):
            with torch.no_grad():
                initial_emb = bert_emb(batch[0].to(device))
                #our_emb = batch[1].to(device)
                #initial_emb = initial_emb + our_emb
                loss, logits = model(inputs_embeds=initial_emb, token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                     extra_emb=batch[1].to(device),
                                     labels=batch[2].to(device))

                total_val_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = batch[2].to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(dev_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(dev_dataloader)
        print('epoch: {a}, acc: {b}'.format(a=epoch, b=avg_val_accuracy))
        if avg_val_accuracy > max_acc:
            max_acc = avg_val_accuracy
            torch.save(model.state_dict(), model_path)


    
    test_acc = list()
    for rate in range(10):
        test_data, test_label = load_data(doc_files[2].format(a=rate*10))
        test_data, test_emb = encode_fn(test_data, embeddings, bert_input)
        test_label = torch.tensor(test_label)
        test_dataloader = DataLoader(TensorDataset(test_data, test_emb, test_label), batch_size=batch_size, shuffle=False)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        total_test_accuracy = 0
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                initial_emb = bert_emb(batch[0].to(device))
                loss, logits = model(inputs_embeds=initial_emb, token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                    extra_emb=batch[1].to(device),
                                    labels=batch[2].to(device))

                logits = logits.detach().cpu().numpy()
                label_ids = batch[2].to('cpu').numpy()
                total_test_accuracy += flat_accuracy(logits, label_ids)
        avg_test_accuracy = total_test_accuracy / len(test_dataloader)
        print(doc_files[2].format(a=rate*10), f'Accuracy: {avg_test_accuracy:.4f}')
        print('\n')
        test_acc.append(avg_test_accuracy)
    return test_acc


def run_sst2():

    # train()
    origin_acc_list = list()
    for lr in [9e-5, 7e-5, 5e-5, 3e-5, 1e-5]:
        print('learning rate = {a}'.format(a=lr))
        acc = train(lr)
        origin_acc_list.append(acc)
        print(acc)
    
    acc_list = list(zip(*origin_acc_list))
    for index, lst in enumerate(acc_list):
        print("typo rate = {a}, acc = {b}".format(a=index*10, b=sum(lst)/len(lst)))
    
    return acc_list



if __name__ == '__main__':
    args = parser.parse_args()
    use_love = args.use_love
    device = torch.device(args.device)
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    max_len = args.max_len
    emb_dim = args.word_embed_dim
    doc_files = [
        '../data/train.tsv',
        '../data/dev.tsv',
        '../data/typo_test_{a}.txt',
    ]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Load the pretrained BERT model
    model = TextClassification.from_pretrained('bert-base-uncased')
    model.to(device)
    bert_emb = model.get_input_embeddings()
    bert_emb.to(device)

    print(run_sst2())



